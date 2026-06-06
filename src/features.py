"""
features.py
-----------
Feature engineering for the Give Me Some Credit dataset.

All four engineered features are derived from domain knowledge about
credit risk.  Weights for the Financial Stress Index are data-driven
(correlation-based) and must be computed on training data only, then
passed through to the test pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Individual feature constructors ──────────────────────────────────────────

def add_total_late_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all three delinquency severity bands into a single count.

    Total_Late_Payments = (30-59 day events)
                        + (60-89 day events)
                        + (90+ day events)

    Rationale: The three columns are highly correlated and together
    represent a borrower's overall payment behaviour.  Summing them
    creates a stronger, more compact signal for tree-based models, and
    is used as the sole delinquency feature for linear models (avoiding
    multicollinearity).
    """
    df = df.copy()
    df["Total_Late_Payments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    return df


def add_income_per_dependent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise monthly income by effective household size.

    Income_Per_Dependent = MonthlyIncome / (NumberOfDependents + 1)

    +1 to denominator: accounts for the borrower and prevents division
    by zero for childless borrowers.
    """
    df = df.copy()
    df["Income_Per_Dependent"] = (
        df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    )
    return df


def compute_fsi_weights(df: pd.DataFrame) -> dict:
    """
    Compute data-driven weights for the Financial Stress Index.

    Weights = normalised absolute Pearson correlations of each component
    against the binary target SeriousDlqin2yrs.

    Must be called on training data ONLY (requires the target column).

    Returns
    -------
    dict with keys: w_debt, w_util, w_late, w_lines
    """
    target = df["SeriousDlqin2yrs"]
    corr_debt  = abs(df["DebtRatio"].corr(target))
    corr_util  = abs(df["RevolvingUtilizationOfUnsecuredLines"].corr(target))
    corr_late  = abs(df["Total_Late_Payments"].corr(target))
    corr_lines = abs(df["NumberOfOpenCreditLinesAndLoans"].corr(target))

    total = corr_debt + corr_util + corr_late + corr_lines
    weights = {
        "w_debt":  corr_debt  / total,
        "w_util":  corr_util  / total,
        "w_late":  corr_late  / total,
        "w_lines": corr_lines / total,
    }

    print("--- FSI Calculated Weights ---")
    print(f"Late Payments Weight: {weights['w_late']:.2f}")
    print(f"Utilization Weight:   {weights['w_util']:.2f}")
    print(f"Debt Ratio Weight:    {weights['w_debt']:.2f}")
    print(f"Open Lines Weight:    {weights['w_lines']:.2f}\n")
    return weights


def add_financial_stress_index(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Compute a correlation-weighted composite Financial Stress Index.

    FSI = w_debt  * DebtRatio
        + w_util  * RevolvingUtilizationOfUnsecuredLines
        + w_late  * Total_Late_Payments
        + w_lines * NumberOfOpenCreditLinesAndLoans

    Parameters
    ----------
    df      : DataFrame (Total_Late_Payments must already exist)
    weights : Output of compute_fsi_weights() from the training set
    """
    df = df.copy()
    df["Financial_Stress_Index"] = (
        weights["w_debt"]  * df["DebtRatio"]
        + weights["w_util"]  * df["RevolvingUtilizationOfUnsecuredLines"]
        + weights["w_late"]  * df["Total_Late_Payments"]
        + weights["w_lines"] * df["NumberOfOpenCreditLinesAndLoans"]
    )
    return df


def compute_disposable_income_bounds(df: pd.DataFrame) -> tuple:
    """
    Compute 1st and 99th percentile clip bounds for Disposable_Income
    from the training set.

    Must be called on training data ONLY so the same bounds are applied
    to the test set without leakage.

    Returns
    -------
    (lower_bound, upper_bound) floats
    """
    raw = df["MonthlyIncome"] - (df["MonthlyIncome"] * df["DebtRatio"])
    return raw.quantile(0.01), raw.quantile(0.99)


def add_disposable_income(df: pd.DataFrame,
                           lower_bound: float,
                           upper_bound: float) -> pd.DataFrame:
    """
    Estimate net monthly disposable income and clip to training-set bounds.

    Disposable_Income = MonthlyIncome - (MonthlyIncome * DebtRatio)
                        clipped to [lower_bound, upper_bound]

    Rationale: DebtRatio represents the fraction of income consumed by
    debt.  A high ratio → negative disposable income → strong default
    signal.  Clipping at 1st/99th percentile prevents extreme outliers
    from distorting the feature without discarding the rows.
    """
    df = df.copy()
    raw = df["MonthlyIncome"] - (df["MonthlyIncome"] * df["DebtRatio"])
    df["Disposable_Income"] = raw.clip(lower=lower_bound, upper=upper_bound)
    return df


# ── Pipeline entry points ─────────────────────────────────────────────────────

def engineer_features_train(df: pd.DataFrame):
    """
    Full feature engineering pipeline for the training set.

    Steps:
      1. Total_Late_Payments
      2. Income_Per_Dependent
      3. Financial_Stress_Index  (weights computed from training data)
      4. Disposable_Income       (clip bounds computed from training data)

    Returns
    -------
    df_eng            : DataFrame with all four new features added
    fsi_weights       : dict — pass to engineer_features_test()
    disposable_bounds : (lower, upper) tuple — pass to engineer_features_test()
    """
    df = add_total_late_payments(df)
    df = add_income_per_dependent(df)

    fsi_weights = compute_fsi_weights(df)
    df = add_financial_stress_index(df, fsi_weights)

    disposable_bounds = compute_disposable_income_bounds(df)
    df = add_disposable_income(df, *disposable_bounds)

    return df, fsi_weights, disposable_bounds


def engineer_features_test(df: pd.DataFrame,
                            fsi_weights: dict,
                            disposable_bounds: tuple) -> pd.DataFrame:
    """
    Full feature engineering pipeline for the test set.
    Uses artefacts computed from the training set — no fitting happens here.

    Parameters
    ----------
    df                : Cleaned test DataFrame
    fsi_weights       : dict from engineer_features_train()
    disposable_bounds : (lower, upper) tuple from engineer_features_train()
    """
    df = add_total_late_payments(df)
    df = add_income_per_dependent(df)
    df = add_financial_stress_index(df, fsi_weights)
    df = add_disposable_income(df, *disposable_bounds)
    return df


# ── Scaling helpers ───────────────────────────────────────────────────────────

# Columns to drop when building the feature matrix for linear models.
# The three original delinquency columns are dropped because Total_Late_Payments
# is their direct sum — including both would cause severe multicollinearity.
LR_DROP_COLS = [
    "SeriousDlqin2yrs", "Id",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def prepare_model_inputs(df_clean: pd.DataFrame):
    """
    Build the two feature matrices used by different model families.

    X_unscaled : All engineered features — for tree-based models.
                 Trees are scale-invariant, so raw values are fine.
                 Retaining the original delinquency columns alongside
                 Total_Late_Payments lets trees exploit both aggregate
                 and per-severity information.

    X_scaled   : Scaled features for linear models (StandardScaler).
                 The three original delinquency columns are dropped to
                 eliminate multicollinearity with Total_Late_Payments.

    Returns
    -------
    X_unscaled, X_scaled, y, scaler
    """
    X_unscaled = df_clean.drop(["SeriousDlqin2yrs", "Id"], axis=1, errors="ignore")
    y = df_clean["SeriousDlqin2yrs"]

    X_linear = df_clean.drop(LR_DROP_COLS, axis=1, errors="ignore")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_linear), columns=X_linear.columns
    )

    print("Unscaled Features (Trees) Shape:", X_unscaled.shape)
    print("Scaled Features (Linear) Shape: ", X_scaled.shape)
    print("Missing values remaining:        ", X_scaled.isnull().sum().sum())

    return X_unscaled, X_scaled, y, scaler
