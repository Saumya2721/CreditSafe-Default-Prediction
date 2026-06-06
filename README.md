# Credit Risk Prediction — Give Me Some Credit (Kaggle)

A complete end-to-end machine learning project for predicting the probability of financial distress (serious delinquency within 2 years) using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle competition dataset.

---

## Results

| Model | ROC-AUC (CV) | Macro-F1 (CV) |
|---|---|---|
| **XGBoost (scale_pos_weight)** ✅ | **0.8644** | 0.6056 |
| XGBoost (SMOTE + Reg) | 0.8535 | 0.6276 |
| Logistic Regression (SMOTE) | 0.8519 | 0.6062 |
| Random Forest (SMOTE) | 0.8517 | 0.6467 |
| Logistic Regression (No SMOTE) | 0.8488 | 0.6119 |

The winning model — **XGBoost with `scale_pos_weight`** — achieved the highest ROC-AUC with 5-fold stratified cross-validation.

---

## Methodology

### 1. Exploratory Data Analysis
- **Missingness audit**: `MonthlyIncome` (19.72% missing), `NumberOfDependents` (2.62% missing)
- **Sentinel value detection**: Values of `96` and `98` in delinquency columns identified as data-entry placeholders (impossible to be 98x late in a 2-year window)
- **Class imbalance**: Only 6.68% of borrowers defaulted — naive all-zero classifier gets 93.32% accuracy but is useless
- **Bivariate analysis**: `NumberOfTimes90DaysLate` showed the strongest default-rate difference across the median (37.02 pp)

### 2. Data Cleaning
- Sentinel values (`96`, `98`) in delinquency columns replaced with `0` and a binary flag (`Sentinel_Flag`) added
- `age == 0` (1 row) treated as data entry error and removed
- `DebtRatio` capped at 5.0, `RevolvingUtilizationOfUnsecuredLines` capped at 2.0 to handle extreme outliers
- `NumberOfDependents` imputed with mode
- `MonthlyIncome` imputed using **KNN Imputer** (k=5) — preserves multivariate structure better than mean/median

### 3. Feature Engineering
Four new features were constructed:

| Feature | Description |
|---|---|
| `Total_Late_Payments` | Sum of all three delinquency columns |
| `Income_Per_Dependent` | Monthly income divided by (dependents + 1) |
| `Financial_Stress_Index` | Correlation-weighted composite of debt, utilisation, late payments, open lines |
| `Disposable_Income` | `MonthlyIncome` × (1 − `DebtRatio`), clipped at 1st–99th percentile |

### 4. Handling Class Imbalance
Two strategies compared head-to-head:
- **SMOTE**: Synthesises new minority class examples by interpolating in feature space → richer decision boundaries
- **`scale_pos_weight`**: Native XGBoost parameter that penalises misclassification of the minority class

`scale_pos_weight` won on ROC-AUC. SMOTE won on Macro-F1 (better minority-class precision).

### 5. Model Development
All models evaluated with **5-fold stratified cross-validation**:
1. **Logistic Regression** (baseline, SMOTE & no-SMOTE variants)
2. **Random Forest** (SMOTE)
3. **XGBoost** (two variants: `scale_pos_weight` and SMOTE + L1/L2 regularisation with `RandomizedSearchCV`)

---

## Setup

```bash
git clone https://github.com/your-username/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
```

Place the raw competition CSVs in `data/raw/`:
- `cs-training.csv`
- `cs-test.csv`

Then run the notebook:
```bash
jupyter notebook notebooks/01_EDA_and_Modeling.ipynb
```

---

## Dependencies

See `requirements.txt`. Key packages:
- `pandas`, `numpy`, `scipy`
- `scikit-learn`, `imbalanced-learn`
- `xgboost`
- `matplotlib`, `seaborn`, `missingno`
