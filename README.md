# Home Loan Approval

End-to-end machine-learning project for the **IBM SkillsBuild Data Analytics with AI Academic Internship Program (BharatCares in association with AICTE)**.

**Author:** Tharun Kumar Varma Chekuri

## Project Description
This project predicts whether a home-loan application should be approved (`Y`) or not approved (`N`) using applicant, income, loan, credit-history and property-area information.

## Problem Statement
Dream Housing Finance wants to automate loan eligibility using customer details collected through an online application. The supplied internship brief asks for EDA, missing-value/outlier treatment, classification models, feature engineering, model comparison and test prediction.

## Dataset
The supplied files are the Analytics Vidhya **Loan Prediction** practice-problem dataset.

Dataset source: https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval/data

The dataset was supplied from the Kaggle **Home Loan Approval** dataset by rishikeshkonapure.

- Training rows: 614
- Training columns: 13
- Test rows: 367
- Test columns: 12
- Target: `Loan_Status`

## Tech Stack
- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Repository Structure
```text
Home-Loan-Approval/
├── README.md
├── requirements.txt
├── .gitignore
├── Tharun_Kumar_Varma_Chekuri_HomeLoanApproval.ipynb
├── Tharun_Kumar_Varma_Chekuri_ProjectReport.docx
├── data/
│   ├── loan_sanction_train.csv
│   └── loan_sanction_test.csv
└── outputs/
    ├── submission.csv
    └── figures/
        ├── 01_target_distribution.png
        ├── 02_categorical_univariate.png
        ├── 03_numerical_univariate.png
        ├── 04_categorical_bivariate.png
        ├── 05_numerical_bivariate.png
        ├── 06_correlation_heatmap.png
        ├── 07_loanamount_log_transform.png
        ├── 08_best_model_confusion_matrix.png
        ├── 09_best_model_roc_curve.png
        └── 10_feature_importance.png
```

## Methodology
1. Problem statement and hypothesis generation
2. Data loading and understanding
3. Univariate and bivariate EDA
4. Missing-value treatment
5. Log transformation for skewed financial variables
6. Stratified validation and five-fold cross-validation
7. Baseline Logistic Regression, Decision Tree and Random Forest
8. Feature engineering: TotalIncome, EMI, BalanceIncome, log transforms
9. RandomizedSearchCV tuning
10. Final refit on all labeled training data
11. Test prediction and `submission.csv`

All imputers and encoders are fitted inside scikit-learn pipelines to avoid data leakage.

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8537 | 0.8317 | 0.9882 | 0.9032 | 0.8498 | 0.7208 |
| Decision Tree | 0.7642 | 0.9000 | 0.7412 | 0.8129 | 0.8204 | 0.6972 |
| **Random Forest** | **0.8618** | **0.8864** | **0.9176** | **0.9017** | **0.8293** | **0.7648** |

**Selected model:** Random Forest

**Best validation metrics:** Accuracy = 0.8618, F1 = 0.9017, ROC-AUC = 0.8293.

> These are held-out validation metrics, not test-set metrics. The supplied test set has no `Loan_Status` labels, so test accuracy cannot be calculated.

## Key Insights
- `Credit_History` is the strongest predictive feature in the fitted Random Forest.
- Training target distribution: 422 approvals (68.73%) and 192 non-approvals (31.27%).
- Credit-history approval rates were 79.6% for `1.0` and 7.9% for `0.0`.
- Semiurban applications had a 76.8% approval rate, compared with 65.8% Urban and 61.5% Rural.
- Log1p reduced measured skew in ApplicantIncome from 6.540 to 0.482 and LoanAmount from 2.678 to -0.150.

## Setup and Run

### 1. Clone
```bash
git clone <my-repo-url>
cd Home-Loan-Approval
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:
```powershell
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Open:
`Tharun_Kumar_Varma_Chekuri_HomeLoanApproval.ipynb`

The notebook reads data using:
```text
data/loan_sanction_train.csv
data/loan_sanction_test.csv
```

## Output
The notebook creates:
```text
outputs/submission.csv
```

with:
```text
Loan_ID,Loan_Status
```

## Academic Note
This project is an academic machine-learning exercise. A real lending system requires additional validation, fairness assessment, governance, explainability, monitoring and human oversight.

## License
Academic project. The supplied dataset remains subject to its original source terms. Code and documentation in this repository may be reused for educational purposes with attribution to the author.
