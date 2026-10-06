"""
Home Loan Approval - Complete VS Code Python Script
Author: Tharun Kumar Varma Chekuri
Program: IBM SkillsBuild Data Analytics with AI Academic Internship Program
Source: https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval/data

Run from VS Code terminal:
    python home_loan_approval.py

Expected repository structure:
Home-Loan-Approval/
├── home_loan_approval.py
├── data/
│   ├── loan_sanction_train.csv
│   └── loan_sanction_test.csv
└── outputs/
    ├── submission.csv
    └── figures/
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)

# -----------------------------------------------------------------------------
# 1. SETTINGS AND PATHS
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "loan_sanction_train.csv"
TEST_PATH = DATA_DIR / "loan_sanction_test.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def save_fig(fig, filename):
    """Save and close a matplotlib figure."""
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path.relative_to(ROOT)}")


def metric_row(y_true, pred, prob):
    """Return the requested classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob),
    }


def engineer_features(df):
    """Create the engineered features required by the project."""
    df = df.copy()

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Illustrative monthly rate = 1% / 12.
    # This is a relative affordability feature, not a contractual interest rate.
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    monthly_rate = 0.01 / 12
    df["EMI"] = (
        df["LoanAmount"] * 1000
        * monthly_rate
        * (1 + monthly_rate) ** term
        / ((1 + monthly_rate) ** term - 1)
    )
    df["BalanceIncome"] = df["TotalIncome"] - df["EMI"]

    for col in [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "TotalIncome",
        "EMI",
        "BalanceIncome",
    ]:
        df["Log_" + col] = np.log1p(df[col].clip(lower=0))

    df["Dependents_num"] = df["Dependents"].replace({"3+": 3}).astype(float)

    # Loan_ID is only used for submission. Raw components are replaced by
    # engineered representations as required by the project workflow.
    df = df.drop(
        columns=[
            "Loan_ID",
            "ApplicantIncome",
            "CoapplicantIncome",
            "Dependents",
        ],
        errors="ignore",
    )
    return df


# -----------------------------------------------------------------------------
# 2. PROBLEM STATEMENT
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("HOME LOAN APPROVAL - END-TO-END MACHINE LEARNING PROJECT")
print("=" * 80)
print("Dream Housing Finance wants to automate home-loan eligibility prediction")
print("using applicant, income, loan, credit-history and property information.")
print("Target: Loan_Status (Y = approved, N = not approved).")

# -----------------------------------------------------------------------------
# 3. HYPOTHESIS GENERATION
# -----------------------------------------------------------------------------
print("\n[2] HYPOTHESES")
print("1. Positive Credit_History should strongly increase approval probability.")
print("2. Higher applicant/household income may improve approval probability.")
print("3. Larger LoanAmount may reduce approval probability.")
print("4. Education may be associated with approval.")
print("5. Property_Area may show different approval rates.")
print("6. Marital status/dependents may be associated with approval.")

# -----------------------------------------------------------------------------
# 4. LOAD DATA
# -----------------------------------------------------------------------------
print("\n[3] LOADING DATA")
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Training shape: {train.shape}")
print(f"Test shape:     {test.shape}")
print("\nFirst five training rows:")
print(train.head().to_string(index=False))

if "Loan_Status" not in train.columns:
    raise ValueError("loan_sanction_train.csv must contain Loan_Status.")
if "Loan_Status" in test.columns:
    raise ValueError("loan_sanction_test.csv should not contain Loan_Status.")

# -----------------------------------------------------------------------------
# 5. UNDERSTANDING THE DATA
# -----------------------------------------------------------------------------
print("\n[4] UNDERSTANDING THE DATA")
print("\nData types:")
print(train.dtypes.to_string())

print("\nDescriptive statistics:")
print(train.describe(include="all").T.to_string())

missing = pd.DataFrame(
    {
        "Train Missing": train.isna().sum(),
        "Train Missing %": (train.isna().mean() * 100).round(2),
        "Test Missing": test.isna().sum(),
        "Test Missing %": (test.isna().mean() * 100).round(2),
    }
)
print("\nMissing values:")
print(missing.to_string())

# -----------------------------------------------------------------------------
# 6. EDA - UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------------
print("\n[5] EXPLORATORY DATA ANALYSIS")
sns.set_theme(style="whitegrid")

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]
numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

print("\nLoan status distribution:")
target_counts = train["Loan_Status"].value_counts()
target_summary = pd.DataFrame(
    {
        "Count": target_counts,
        "Percent": (target_counts / len(train) * 100).round(2),
    }
)
print(target_summary.to_string())

fig, ax = plt.subplots(figsize=(6, 4))
target_counts.sort_index().plot(kind="bar", ax=ax)
ax.set_title("Loan Status Distribution")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")
save_fig(fig, "01_target_distribution.png")

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.ravel()
for i, col in enumerate(categorical_cols):
    train[col].value_counts(dropna=False).plot(kind="bar", ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis="x", rotation=25)
for j in range(len(categorical_cols), len(axes)):
    axes[j].axis("off")
save_fig(fig, "02_categorical_univariate.png")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), numerical_cols):
    ax.hist(train[col].dropna(), bins=25)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
save_fig(fig, "03_numerical_univariate.png")

print("\nNumerical summary:")
print(train[numerical_cols].describe().T.to_string())

# -----------------------------------------------------------------------------
# 7. EDA - BIVARIATE ANALYSIS
# -----------------------------------------------------------------------------
print("\nApproval rates by categorical variables:")
for col in categorical_cols:
    table = pd.crosstab(train[col], train["Loan_Status"], normalize="index") * 100
    print(f"\n{col}:")
    print(table.round(2).to_string())

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()
for i, col in enumerate(categorical_cols):
    rate = (
        train.groupby(col)["Loan_Status"]
        .apply(lambda s: (s == "Y").mean() * 100)
        .sort_values(ascending=False)
    )
    axes[i].bar(rate.index.astype(str), rate.values)
    axes[i].set_title(f"Approval Rate by {col}")
    axes[i].set_ylabel("Approval Rate (%)")
    axes[i].tick_params(axis="x", rotation=25)
for j in range(len(categorical_cols), len(axes)):
    axes[j].axis("off")
save_fig(fig, "04_categorical_bivariate.png")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), numerical_cols):
    sns.boxplot(data=train, x="Loan_Status", y=col, ax=ax)
    ax.set_title(f"{col} vs Loan Status")
save_fig(fig, "05_numerical_bivariate.png")

corr_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
corr = train[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap of Numerical Features")
save_fig(fig, "06_correlation_heatmap.png")

# -----------------------------------------------------------------------------
# 8. MISSING VALUES AND OUTLIER / SKEW TREATMENT
# -----------------------------------------------------------------------------
print("\n[6] MISSING VALUE AND SKEW TREATMENT")
print("Categorical variables: most frequent value (mode) inside pipeline.")
print("Numerical variables: median inside pipeline.")
print("Financial skew: log1p transformation; observations are retained.")

skew_before = train[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]].skew()
log_demo = train[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]].copy()
for col in log_demo.columns:
    log_demo[col] = np.log1p(log_demo[col].clip(lower=0))
skew_after = log_demo.skew()
skew_summary = pd.DataFrame(
    {"Skew Before": skew_before, "Skew After log1p": skew_after}
).round(3)
print("\nSkewness before and after log1p:")
print(skew_summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train["LoanAmount"].dropna(), bins=25)
axes[0].set_title("LoanAmount Before log1p")
axes[1].hist(np.log1p(train["LoanAmount"].dropna().clip(lower=0)), bins=25)
axes[1].set_title("LoanAmount After log1p")
save_fig(fig, "07_loanamount_log_transform.png")

# -----------------------------------------------------------------------------
# 9. EVALUATION METRICS
# -----------------------------------------------------------------------------
print("\n[7] EVALUATION METRICS")
print("Accuracy   : all predictions correct / all predictions")
print("Precision  : correct predicted approvals / predicted approvals")
print("Recall     : correct predicted approvals / actual approvals")
print("F1-score   : harmonic mean of precision and recall")
print("ROC-AUC    : discrimination across classification thresholds")
print("Confusion matrix: true/false positive and negative counts")

# -----------------------------------------------------------------------------
# 10. TARGET / FEATURES
# -----------------------------------------------------------------------------
y = (train["Loan_Status"] == "Y").astype(int)
X = train.drop(columns=["Loan_Status"])
X_test_original = test.copy()

print(f"\nApproval rate (Y): {y.mean():.4f}")
print(f"Non-approval rate (N): {1 - y.mean():.4f}")

# -----------------------------------------------------------------------------
# 11. MODEL BUILDING PART 1 - BASELINE MODELS
# -----------------------------------------------------------------------------
print("\n[8] MODEL BUILDING PART 1 - BASELINE MODELS")

baseline_features = [c for c in X.columns if c != "Loan_ID"]
baseline_num = [c for c in baseline_features if X[c].dtype != "object"]
baseline_cat = [c for c in baseline_features if X[c].dtype == "object"]

baseline_pre = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            baseline_num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            baseline_cat,
        ),
    ]
)

baseline_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
    ),
}

X_train_b, X_valid_b, y_train_b, y_valid_b = train_test_split(
    X[baseline_features],
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

baseline_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)

baseline_rows = []
for name, model in baseline_models.items():
    pipe = Pipeline([("preprocessor", baseline_pre), ("model", model)])
    pipe.fit(X_train_b, y_train_b)
    pred = pipe.predict(X_valid_b)
    prob = pipe.predict_proba(X_valid_b)[:, 1]
    m = metric_row(y_valid_b, pred, prob)

    cv_scores = cross_validate(
        pipe,
        X[baseline_features],
        y,
        cv=baseline_cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=1,
    )
    m.update(
        {
            "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
            "CV Precision Mean": cv_scores["test_precision"].mean(),
            "CV Recall Mean": cv_scores["test_recall"].mean(),
            "CV F1 Mean": cv_scores["test_f1"].mean(),
            "CV ROC-AUC Mean": cv_scores["test_roc_auc"].mean(),
        }
    )
    m["Model"] = name
    baseline_rows.append(m)

baseline_results = pd.DataFrame(baseline_rows).set_index("Model")
print("\nBaseline validation and CV results:")
print(baseline_results.round(4).to_string())

# -----------------------------------------------------------------------------
# 12. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print("\n[9] FEATURE ENGINEERING")
print("TotalIncome = ApplicantIncome + CoapplicantIncome")
print("EMI = estimated monthly payment using an illustrative 1%/12 rate")
print("BalanceIncome = TotalIncome - EMI")
print("Log_* = log1p financial transformations")
print("Dependents_num maps 3+ to 3")

X_engineered = engineer_features(X)
X_test_engineered = engineer_features(X_test_original)

print(f"Engineered feature count: {X_engineered.shape[1]}")
print("Engineered columns:")
print(list(X_engineered.columns))

# -----------------------------------------------------------------------------
# 13. MODEL BUILDING PART 2 - TUNING
# -----------------------------------------------------------------------------
print("\n[10] MODEL BUILDING PART 2 - HYPERPARAMETER TUNING")

eng_num = [c for c in X_engineered.columns if X_engineered[c].dtype != "object"]
eng_cat = [c for c in X_engineered.columns if X_engineered[c].dtype == "object"]

eng_pre = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            eng_num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            eng_cat,
        ),
    ]
)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear"],
        },
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "model__max_depth": [2, 3, 4, 5, 6, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 4, 6, 8],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        },
    ),
}

X_train, X_valid, y_train, y_valid = train_test_split(
    X_engineered,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

tuned_rows = []
tuned_models = {}

for name, (model, params) in models.items():
    print(f"\nTuning {name} ...")
    pipe = Pipeline([("preprocessor", eng_pre), ("model", model)])
    n_total = int(np.prod([len(v) for v in params.values()]))

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=min(4, n_total),
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    pred = search.predict(X_valid)
    prob = search.predict_proba(X_valid)[:, 1]
    m = metric_row(y_valid, pred, prob)
    m.update(
        {
            "CV ROC-AUC Mean": search.best_score_,
            "Best Params": str(search.best_params_),
            "Model": name,
        }
    )
    tuned_rows.append(m)
    tuned_models[name] = search

comparison = pd.DataFrame(tuned_rows).set_index("Model")
print("\nTuned model comparison:")
print(
    comparison[
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC Mean"]
    ]
    .round(4)
    .to_string()
)

# Select the final model using the CV ROC-AUC criterion.
best_name = comparison["CV ROC-AUC Mean"].idxmax()
best_search = tuned_models[best_name]
print(f"\nSelected best model: {best_name}")
print(f"Best parameters: {best_search.best_params_}")

# -----------------------------------------------------------------------------
# 14. BEST MODEL EVALUATION
# -----------------------------------------------------------------------------
print("\nBEST MODEL VALIDATION")
best_pred = best_search.predict(X_valid)
best_prob = best_search.predict_proba(X_valid)[:, 1]
best_metrics = metric_row(y_valid, best_pred, best_prob)

for key, value in best_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nClassification report:")
print(classification_report(y_valid, best_pred, target_names=["N", "Y"]))

cm = confusion_matrix(y_valid, best_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["N", "Y"]).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title(f"Confusion Matrix - {best_name}")
save_fig(fig, "08_best_model_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_valid, best_prob)
auc_value = roc_auc_score(y_valid, best_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.3f}")
ax.plot([0, 1], [0, 1], "--", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve - {best_name}")
ax.legend()
save_fig(fig, "09_best_model_roc_curve.png")

# -----------------------------------------------------------------------------
# 15. FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
print("\nFEATURE IMPORTANCE")
fitted_pre = best_search.best_estimator_.named_steps["preprocessor"]
fitted_model = best_search.best_estimator_.named_steps["model"]
feature_names = fitted_pre.get_feature_names_out()

if hasattr(fitted_model, "coef_"):
    importance = np.abs(fitted_model.coef_[0])
else:
    importance = fitted_model.feature_importances_

fi = pd.DataFrame({"Feature": feature_names, "Importance": importance})
fi = fi.sort_values("Importance", ascending=False).head(15)
print(fi.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
fi_sorted = fi.sort_values("Importance")
ax.barh(fi_sorted["Feature"], fi_sorted["Importance"])
ax.set_title(f"Top Feature Importance - {best_name}")
ax.set_xlabel("Importance")
save_fig(fig, "10_feature_importance.png")

# -----------------------------------------------------------------------------
# 16. FINAL FIT ON ALL LABELED DATA + TEST PREDICTIONS
# -----------------------------------------------------------------------------
print("\n[11] FINAL TEST PREDICTION")
final_model = best_search.best_estimator_
final_model.fit(X_engineered, y)

test_pred_num = final_model.predict(X_test_engineered)
test_pred = np.where(test_pred_num == 1, "Y", "N")

submission = pd.DataFrame(
    {
        "Loan_ID": test["Loan_ID"],
        "Loan_Status": test_pred,
    }
)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved: {SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print("\nFirst 10 predictions:")
print(submission.head(10).to_string(index=False))
print("\nPrediction distribution:")
print(submission["Loan_Status"].value_counts().to_string())

# -----------------------------------------------------------------------------
# 17. FINAL SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL PROJECT SUMMARY")
print("=" * 80)
print(f"Best Model : {best_name}")
for key, value in best_metrics.items():
    print(f"{key:<10}: {value:.4f}")
print(f"CV ROC-AUC : {comparison.loc[best_name, 'CV ROC-AUC Mean']:.4f}")
print(f"Submission : {SUBMISSION_PATH.relative_to(ROOT)}")
print(f"Figures    : {FIG_DIR.relative_to(ROOT)}")
print("=" * 80)
print("Project completed successfully.")
