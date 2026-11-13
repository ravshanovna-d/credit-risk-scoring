"""Train the credit risk model, evaluate it, and export it for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
import time
import warnings
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from export_utils import check_onnx, examples, export_onnx, feature_schema, write_meta

ROOT = Path(__file__).parent
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
SEED = 42
COST_FN, COST_FP = 5, 1   # approving a bad loan costs 5x more than rejecting a good applicant

COLUMNS = ["status", "duration", "credit_history", "purpose", "amount", "savings",
           "employment_duration", "installment_rate", "personal_status_sex", "other_debtors",
           "present_residence", "property", "age", "other_installment_plans", "housing",
           "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk"]
# Attribute codes from the UCI documentation (german.doc). DM = Deutsche Mark.
CODES = {
    "status": {"A11": "< 0 DM", "A12": "0–200 DM", "A13": "≥ 200 DM / salary account", "A14": "No checking account"},
    "credit_history": {"A30": "No credits / all paid back", "A31": "All credits at this bank paid back",
                       "A32": "Existing credits paid on time", "A33": "Delays in the past",
                       "A34": "Critical account / credits elsewhere"},
    "purpose": {"A40": "New car", "A41": "Used car", "A42": "Furniture / equipment", "A43": "Radio / TV",
                "A44": "Domestic appliances", "A45": "Repairs", "A46": "Education", "A47": "Vacation",
                "A48": "Retraining", "A49": "Business", "A410": "Other"},
    "savings": {"A61": "< 100 DM", "A62": "100–500 DM", "A63": "500–1000 DM", "A64": "≥ 1000 DM",
                "A65": "Unknown / no savings"},
    "employment_duration": {"A71": "Unemployed", "A72": "< 1 year", "A73": "1–4 years",
                            "A74": "4–7 years", "A75": "≥ 7 years"},
    "personal_status_sex": {"A91": "Male: divorced/separated", "A92": "Female: divorced/separated/married",
                            "A93": "Male: single", "A94": "Male: married/widowed", "A95": "Female: single"},
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {"A121": "Real estate", "A122": "Savings agreement / life insurance",
                 "A123": "Car or other", "A124": "Unknown / none"},
    "other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "job": {"A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
            "A173": "Skilled employee / official", "A174": "Management / self-employed / highly qualified"},
    "telephone": {"A191": "No", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}
# The 13 features used by the demo (personal status / sex and foreign-worker status are left out on purpose).
NUM_COLS = ["duration", "amount", "age", "installment_rate", "number_credits"]
CAT_COLS = ["status", "credit_history", "purpose", "savings", "employment_duration",
            "property", "other_installment_plans", "housing"]
LABELS = {
    "status": "Checking account balance", "duration": "Loan duration", "credit_history": "Credit history",
    "purpose": "Purpose", "amount": "Loan amount", "savings": "Savings",
    "employment_duration": "Employed for", "installment_rate": "Installment rate",
    "property": "Most valuable property", "age": "Age", "other_installment_plans": "Other installment plans",
    "housing": "Housing", "number_credits": "Existing credits at this bank",
}
HELPS = {"duration": "months", "amount": "DM", "age": "years",
         "installment_rate": "1–4, higher = larger share of income", "number_credits": "count"}


def fetch(url: str, dest: Path, tries: int = 4) -> Path:
    """Download once into data/ (retries for flaky connections)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return dest
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(fetch(DATA_URL, ROOT / "data" / "german.data"), sep=" ", header=None, names=COLUMNS)
    for col, mapping in CODES.items():
        df[col] = df[col].map(mapping)
    df["bad_risk"] = (df.pop("credit_risk") == 2).astype(int)   # 1 = good, 2 = bad in the raw file
    for c in NUM_COLS:
        df[c] = df[c].astype(float)
    return df


def make_prep(scale: bool) -> ColumnTransformer:
    """Scaling matters for LR / SVC; tree ensembles get the raw numbers."""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def build_models(cv_inner):
    prep = make_prep(scale=True)
    svc_grid = [
        {"model__estimator__kernel": ["rbf"], "model__estimator__C": [0.1, 0.3, 1, 3, 10],
         "model__estimator__gamma": ["scale", 0.01, 0.03]},
        {"model__estimator__kernel": ["linear"], "model__estimator__C": [0.01, 0.1, 1]},
    ]
    return {
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=0.3))]),
        "SVC (tuned)": GridSearchCV(
            # SVC scores -> probabilities with Platt scaling (sigmoid), fitted on internal CV folds
            Pipeline([("prep", prep), ("model", CalibratedClassifierCV(
                SVC(), method="sigmoid", cv=5, ensemble=False))]),
            svc_grid, cv=cv_inner, scoring="roc_auc", n_jobs=2),
        "Random Forest": Pipeline([("prep", make_prep(scale=False)), ("model", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", make_prep(scale=False)), ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED))]),
    }


def avg_cost(y_true, proba, t):
    """Average cost per applicant when everyone with P(bad) >= t is rejected."""
    tn, fp, fn, tp = confusion_matrix(y_true, (proba >= t).astype(int), labels=[0, 1]).ravel()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)


def main():
    # skl2onnx reads SVC attributes that scikit-learn 1.9 marks as deprecated; harmless here.
    warnings.filterwarnings("ignore", category=FutureWarning)
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df[NUM_COLS + CAT_COLS], df["bad_risk"]
    print(f"{len(df)} applicants, {y.mean():.1%} bad risk")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1) Compare models with 5-fold CV (SVC tuned inside each fold = nested CV)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    models = build_models(cv_inner)
    rows = []
    for name, model in models.items():
        r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=2,
                           scoring={"ROC-AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"})
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in ("ROC-AUC", "F1", "Accuracy")}})
        print(f"{name:22s} ROC-AUC={rows[-1]['ROC-AUC']:.3f}  F1={rows[-1]['F1']:.3f}  Acc={rows[-1]['Accuracy']:.3f}")
    comparison = sorted(rows, key=lambda r: -r["ROC-AUC"])
    # Prefer the simplest model (dict order = simplest first) within 0.005 ROC-AUC of the top score.
    top = comparison[0]["ROC-AUC"]
    best_name = next(r["model"] for r in rows if r["ROC-AUC"] >= top - 0.005)
    best = models[best_name]
    if best_name.startswith("SVC"):
        best = best.fit(X_train, y_train).best_estimator_
        print("Best SVC:", best.named_steps["model"].estimator)
    else:
        best = best.fit(X_train, y_train)

    # 2) Cost-sensitive threshold, chosen on out-of-fold training predictions (not on the test set)
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=2)[:, 1]
    grid_t = np.round(np.arange(0.05, 0.81, 0.01), 2)
    oof_cost = [avg_cost(y_train, oof, t) for t in grid_t]
    cost_t = float(grid_t[int(np.argmin(oof_cost))])
    print(f"Cost-optimal threshold (OOF): {cost_t:.2f}  (theory for calibrated probabilities: {COST_FP / (COST_FP + COST_FN):.2f})")

    # 3) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    pred_c = (proba >= cost_t).astype(int)
    cost_05, cost_c = avg_cost(y_test, proba, 0.5), avg_cost(y_test, proba, cost_t)
    cost_all_good = avg_cost(y_test, proba, 1.01)   # approve everyone
    test_metrics = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Recall (bad) @ 0.5": recall_score(y_test, pred),
        f"Recall (bad) @ {cost_t:.2f}": recall_score(y_test, pred_c),
        f"Precision (bad) @ {cost_t:.2f}": precision_score(y_test, pred_c),
        "Cost / applicant @ 0.5": cost_05,
        f"Cost / applicant @ {cost_t:.2f}": cost_c,
    }
    extra = {"F1 @ 0.5": f1_score(y_test, pred), "Precision (bad) @ 0.5": precision_score(y_test, pred),
             "Accuracy @ cost threshold": accuracy_score(y_test, pred_c), "Cost / applicant, approve all": cost_all_good}
    print("Test:", {k: round(v, 3) for k, v in {**test_metrics, **extra}.items()})

    # 4) Figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred_c, display_labels=["Good", "Bad"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix (test set, threshold {cost_t:.2f})")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "roc_curve.png", dpi=130); plt.close(fig)

    test_cost = [avg_cost(y_test, proba, t) for t in grid_t]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid_t, oof_cost, label="Cross-validation (train)", color="#4f46e5")
    ax.plot(grid_t, test_cost, label="Test set", color="#f59e0b")
    ax.axvline(cost_t, ls="--", color="grey"); ax.axvline(0.5, ls=":", color="grey")
    ax.text(cost_t + 0.01, max(oof_cost) * 0.95, f"chosen {cost_t:.2f}", fontsize=9)
    ax.set_xlabel("Reject if P(bad) ≥ threshold"); ax.set_ylabel("Average cost per applicant")
    ax.set_title(f"Business cost (bad loan = {COST_FN}× a rejected good one)"); ax.legend()
    fig.tight_layout(); fig.savefig(ASSETS / "cost_vs_threshold.png", dpi=130); plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=2)
    imp = pd.Series(imp.importances_mean, index=[LABELS[c] for c in X_test.columns]).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Most important features")
    fig.tight_layout(); fig.savefig(ASSETS / "feature_importance.png", dpi=130); plt.close(fig)

    rate = df.groupby("status")["bad_risk"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rate.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Share of bad credit risks"); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Bad-risk rate by checking account balance")
    fig.tight_layout(); fig.savefig(ASSETS / "bad_rate_by_status.png", dpi=130); plt.close(fig)

    # 5) Export for the browser and verify it matches scikit-learn
    export_onnx(best, X_train, WEB / "model.onnx")
    err = check_onnx(WEB / "model.onnx", best, X_test)
    print(f"ONNX check OK (max diff {err:.2e})")

    write_meta(WEB / "meta.json", {
        "title": "Credit Risk Scoring",
        "subtitle": ("Scores a loan application by the probability that the borrower turns out to be a "
                     "bad credit risk, with a decision threshold tuned for the real cost of a bad loan."),
        "tags": ["Classification", "scikit-learn", best_name, "Cost-sensitive threshold", "Nested CV", "ONNX"],
        "task": "classification",
        "classes": [0, 1], "positive_class": 1,
        "output_label": "Probability of bad credit risk",
        "verdicts": ["Approve — low risk", "Review — above the cost-based cut-off", "Likely reject — high risk"],
        "thresholds": [cost_t, 0.5],
        "result_note": (f"A bad loan is assumed to cost {COST_FN}× more than turning away a good customer, so "
                        f"applications are flagged from {cost_t:.0%} risk, not 50%. Educational demo on 1970s German data — "
                        "not a real lending decision."),
        "features": feature_schema(X_train, LABELS, HELPS),
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {k: "percent" for k in test_metrics if k.startswith(("Accuracy", "Recall", "Precision"))},
        "metrics_note": (f"Held-out test set of {len(y_test)} applicants. Cost = {COST_FN} per approved bad loan + "
                         f"{COST_FP} per rejected good applicant, averaged per applicant (approving everyone costs "
                         f"{cost_all_good:.2f})."),
        "figures": [
            {"src": "assets/cost_vs_threshold.png", "caption": "Choosing the cut-off by business cost instead of using 0.5."},
            {"src": "assets/feature_importance.png", "caption": "What drives the prediction the most."},
            {"src": "assets/bad_rate_by_status.png", "caption": "Applicants with a negative checking balance default most often."},
            {"src": "assets/roc_curve.png", "caption": "ROC curve on the test set."},
            {"src": "assets/confusion_matrix.png", "caption": f"Test-set decisions at the cost-based threshold ({cost_t:.2f})."},
        ],
        "steps": [
            f"Loaded {len(df):,} loan applications (UCI Statlog German Credit) and decoded the A11/A12… "
            "attribute codes into readable categories.",
            "Kept 13 features a loan officer would enter; left out sex / marital status and foreign-worker status.",
            "Built one scikit-learn Pipeline: scaling for numbers, one-hot encoding for categories.",
            "Compared Logistic Regression, a tuned SVC (grid search inside each fold), Random Forest and "
            "Gradient Boosting with stratified 5-fold cross-validation.",
            f"Picked {best_name} by ROC-AUC, then chose the decision threshold that minimises expected cost "
            f"(bad loan = {COST_FN}× a rejected good one) on out-of-fold predictions.",
            "Evaluated once on a held-out 20% test set and exported the pipeline to ONNX for the browser.",
        ],
        "dataset": {"name": "Statlog (German Credit Data), UCI", "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data", "rows": len(df)},
        "repo_url": "https://github.com/ravshanovna-d/credit-risk-scoring",
        "notebook_url": "https://github.com/ravshanovna-d/credit-risk-scoring/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    main()
