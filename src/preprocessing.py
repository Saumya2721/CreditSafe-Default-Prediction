"""
preprocessing.py
----------------
Data cleaning and imputation for the Give Me Some Credit dataset.

Fittable objects (KNN imputer, scaler) are always fit on training data only
and applied to test data via transform() to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


SENTINEL_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Exactly the four features used in the notebook's KNN imputation block
KNN_IMPUTE_FEATURES = [
    "age",
    "NumberRealEstateLoansOrLines",
    "NumberOfOpenCreditLinesAndLoans",
    "MonthlyIncome",
]


# ── Individual cleaning steps ─────────────────────────────────────────────────

def flag_and_clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Sentinel_Flag (1 = row had a 96/98 in any delinquency column),
    then replace 96 and 98 with 0 across all three delinquency columns.
    The flag is added BEFORE the values are altered so the signal is preserved.
    """
    df = df.copy()
    df["Sentinel_Flag"] = (
        df[SENTINEL_COLS].isin([96, 98]).any(axis=1).astype(int)
    )
    for col in SENTINEL_COLS:
        df[col] = df[col].replace([96, 98], 0)
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers:
      - RevolvingUtilizationOfUnsecuredLines → max 2.0
      - DebtRatio → max 5.0
    """
    df = df.copy()
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=2.0)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=5.0)
    return df


def fix_age_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace age == 0 (data entry error, 1 row in training set) with the
    column median.
    """
    df = df.copy()
    df["age"] = df["age"].replace(0, df["age"].median())
    return df


def flag_missing_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MonthlyIncome_missing (1 = was NaN) BEFORE imputation so the model
    can learn from the fact that income was not reported.
    """
    df = df.copy()
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isnull().astype(int)
    return df


def impute_dependents(df: pd.DataFrame,
                      mode_value: float = None) -> pd.DataFrame:
    """
    Fill NaNs in NumberOfDependents with the mode (0 in practice).

    Parameters
    ----------
    df         : DataFrame
    mode_value : Pre-computed mode from training data.  If None, computes
                 from df (training path only).
    """
    df = df.copy()
    if mode_value is None:
        mode_value = df["NumberOfDependents"].mode()[0]
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(mode_value)
    return df


class IncomeImputer:
    """
    KNN imputer for MonthlyIncome.

    Internally fits a StandardScaler → KNNImputer(k=5) on the four
    KNN_IMPUTE_FEATURES columns.  The scaler is needed because KNN is
    distance-based and the raw features span very different magnitudes.

    Usage
    -----
    imputer = IncomeImputer().fit(train_df)
    train_df = imputer.transform(train_df)
    test_df  = imputer.transform(test_df)   # uses the same fitted objects
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._scaler  = StandardScaler()
        self._imputer = KNNImputer(n_neighbors=n_neighbors)
        self._income_idx = KNN_IMPUTE_FEATURES.index("MonthlyIncome")

    def fit(self, df: pd.DataFrame) -> "IncomeImputer":
        subset = df[KNN_IMPUTE_FEATURES]
        scaled = self._scaler.fit_transform(subset)
        self._imputer.fit(scaled)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        subset = df[KNN_IMPUTE_FEATURES]
        scaled          = self._scaler.transform(subset)
        imputed_scaled  = self._imputer.transform(scaled)
        imputed         = self._scaler.inverse_transform(imputed_scaled)
        df["MonthlyIncome"] = imputed[:, self._income_idx]
        return df


# ── Pipeline entry points ─────────────────────────────────────────────────────

def clean_train(df: pd.DataFrame):
    """
    Full cleaning pipeline for the training set.

    Steps (in order):
      1. Flag + replace sentinel values in delinquency columns
      2. Cap extreme outliers in RevolvingUtilization and DebtRatio
      3. Replace age == 0 with column median
      4. Add MonthlyIncome_missing flag
      5. Impute NumberOfDependents with mode
      6. Fit IncomeImputer and impute MonthlyIncome via KNN

    Returns
    -------
    df_clean       : Cleaned DataFrame
    income_imputer : Fitted IncomeImputer — pass to clean_test()
    mode_dep       : Mode of NumberOfDependents — pass to clean_test()
    """
    df = flag_and_clean_sentinels(df)
    df = cap_outliers(df)
    df = fix_age_zero(df)
    df = flag_missing_income(df)

    mode_dep = df["NumberOfDependents"].mode()[0]
    df = impute_dependents(df, mode_value=mode_dep)

    income_imputer = IncomeImputer(n_neighbors=5).fit(df)
    df = income_imputer.transform(df)

    return df, income_imputer, mode_dep


def clean_test(df: pd.DataFrame,
               income_imputer: IncomeImputer,
               mode_dep: float) -> pd.DataFrame:
    """
    Full cleaning pipeline for the test set.
    Uses artefacts fit on the training set — no fitting happens here.

    Parameters
    ----------
    df             : Raw test DataFrame
    income_imputer : Fitted IncomeImputer from clean_train()
    mode_dep       : Mode of NumberOfDependents from clean_train()
    """
    df = flag_and_clean_sentinels(df)
    df = cap_outliers(df)
    df = flag_missing_income(df)
    df = impute_dependents(df, mode_value=mode_dep)
    df = income_imputer.transform(df)
    return df
