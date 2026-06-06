"""
eda.py
------
Exploratory Data Analysis functions for the Give Me Some Credit dataset.

Each function performs one clearly scoped analysis and either prints results
or returns a figure/DataFrame so the notebook stays readable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import missingno as msno
from scipy import stats


# ── 1. Missingness Audit ──────────────────────────────────────────────────────

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of missing-value counts and percentages,
    sorted from most to least missing.
    """
    summary = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2),
    }).sort_values("Missing Count", ascending=False)
    print("Missing Value Summary:")
    print(summary)
    return summary


def plot_missingno(df: pd.DataFrame) -> None:
    """
    Side-by-side missingno matrix and bar chart showing co-occurrence
    and proportion of missingness per column.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    msno.matrix(df, ax=axes[0], sparkline=False, fontsize=10)
    axes[0].set_title("Missingno Matrix")
    msno.bar(df, ax=axes[1], fontsize=10, color="steelblue")
    axes[1].set_title("Missingno Bar Chart")
    plt.show()


def plot_missingness_heatmap(df: pd.DataFrame, sample_n: int = 10_000) -> None:
    """
    Heatmap of the missingness indicator matrix, rows sorted by total
    missing count.  A sample is used for rendering speed.
    """
    miss_indicator = df.isnull().astype(int)
    miss_indicator["total_missing"] = miss_indicator.sum(axis=1)
    miss_sorted = (
        miss_indicator.sort_values("total_missing", ascending=False)
        .drop("total_missing", axis=1)
    )
    miss_sample = miss_sorted.iloc[:sample_n]

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        miss_sample, cbar=True, cmap="YlOrRd", yticklabels=False,
        linewidths=0, cbar_kws={"label": "1 = Missing, 0 = Observed"},
    )
    plt.title(
        f"Missingness Indicator Heatmap — {sample_n:,} rows sorted by missingness per row\n"
        "(Horizontal bands = rows with same features missing together)",
        fontsize=12, fontweight="bold",
    )
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.show()

    both = (df["MonthlyIncome"].isnull() & df["NumberOfDependents"].isnull()).sum()
    expected = (
        df["MonthlyIncome"].isnull().mean()
        * df["NumberOfDependents"].isnull().mean()
        * len(df)
    )
    print(f"Both MonthlyIncome AND NumberOfDependents missing: {both:,} rows")
    print(f"Expected if independent (MCAR): {expected:.0f} rows")
    print(f"Ratio (actual/expected): {both/expected:.2f}x")


def plot_income_missing_distributions(df: pd.DataFrame) -> None:
    """
    KDE plots comparing feature distributions between rows where
    MonthlyIncome is observed vs missing.  Different curves are evidence
    against MCAR (completely random missingness).
    """
    compare_cols = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines",
        "SeriousDlqin2yrs",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    miss_mask = df["MonthlyIncome"].isnull()

    for i, col in enumerate(compare_cols):
        clip_val = df[col].quantile(0.99)
        obs_vals  = df.loc[~miss_mask, col].clip(upper=clip_val).dropna()
        miss_vals = df.loc[ miss_mask, col].clip(upper=clip_val).dropna()
        obs_vals.plot(kind="kde", ax=axes[i], label="MonthlyIncome observed",
                      color="steelblue", linewidth=2)
        miss_vals.plot(kind="kde", ax=axes[i], label="MonthlyIncome missing",
                       color="crimson", linewidth=2, linestyle="--")
        axes[i].set_title(col, fontweight="bold", fontsize=10)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel("")

    plt.suptitle(
        "Feature Distributions: Rows where MonthlyIncome is Observed vs Missing\n"
        "(Different curves = missingness is NOT random = evidence against MCAR)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()


def run_little_mcar_test(df: pd.DataFrame) -> None:
    """
    Run Little's MCAR test.  Requires pyampute (pip install pyampute).

    H₀: Data is Missing Completely At Random.
    Reject H₀ if p < 0.05 → data is MAR or MNAR.
    """
    from pyampute.exploration.mcar_statistical_tests import MCARTest

    cols_for_mcar = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
    ]
    mcar_tester = MCARTest(method="little")
    little_p = mcar_tester.little_mcar_test(df[cols_for_mcar])

    print("Little's MCAR Test")
    print("=" * 40)
    p_display = (
        f"{little_p:.6f}" if little_p > 1e-9 else f"{little_p:.2e} (effectively 0)"
    )
    print(f"p-value : {p_display}")
    result = (
        "REJECT H0 -- data is NOT MCAR"
        if little_p < 0.05
        else "Fail to reject H0 -- consistent with MCAR"
    )
    print(f"Decision: {result}")


def point_biserial_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute point-biserial correlations between the MonthlyIncome_missing
    indicator and every continuous predictor.

    Significant correlations → missingness is predictable from observed
    features → MAR (Missing At Random).

    Returns a sorted DataFrame of results.
    """
    df = df.copy()
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isnull().astype(int)
    n_missing = df["MonthlyIncome_missing"].sum()
    print(
        f"MonthlyIncome_missing: {n_missing:,} missing rows "
        f"({n_missing / len(df) * 100:.2f}%)\n"
    )

    continuous_predictors = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
        "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
        "SeriousDlqin2yrs",
    ]
    corr_results = []
    for col in continuous_predictors:
        valid = df[["MonthlyIncome_missing", col]].dropna()
        r, p = stats.pointbiserialr(valid["MonthlyIncome_missing"], valid[col])
        corr_results.append({
            "Feature": col,
            "r (point-biserial)": round(r, 4),
            "p-value": f"{p:.2e}",
            "Sig at p<0.01": "YES" if p < 0.01 else "NO",
            "Strength": (
                "Strong" if abs(r) > 0.3 else "Moderate" if abs(r) > 0.1 else "Weak"
            ),
        })

    corr_df = (
        pd.DataFrame(corr_results)
        .sort_values("r (point-biserial)", key=abs, ascending=False)
    )
    print("Point-Biserial Correlations — MonthlyIncome_missing vs each feature:")
    return corr_df


def plot_mar_evidence(df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    """
    Two-panel plot: (1) bar chart of point-biserial correlations,
    (2) missingness rate by age group.

    Parameters
    ----------
    df      : Raw training DataFrame (before adding MonthlyIncome_missing)
    corr_df : Output of point_biserial_missingness()
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    colors = ["#E74C3C" if r > 0 else "#2980B9" for r in corr_df["r (point-biserial)"]]
    axes[0].barh(corr_df["Feature"], corr_df["r (point-biserial)"],
                 color=colors, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].axvline( 0.1, color="gray", linewidth=0.6, linestyle="--",
                    alpha=0.6, label="|r|=0.1 (weak)")
    axes[0].axvline(-0.1, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    axes[0].axvline( 0.3, color="orange", linewidth=0.6, linestyle="--",
                    alpha=0.8, label="|r|=0.3 (strong)")
    axes[0].set_xlabel("Point-Biserial Correlation with MonthlyIncome_missing")
    axes[0].set_title(
        "Point-Biserial Correlations\n"
        "(Red = positive correlation with missingness, Blue = negative)",
        fontweight="bold",
    )
    axes[0].legend(fontsize=8)

    df_tmp = df.copy()
    df_tmp["MonthlyIncome_missing"] = df_tmp["MonthlyIncome"].isnull().astype(int)
    df_tmp["age_bin"] = pd.cut(
        df_tmp["age"], bins=[0, 30, 40, 50, 60, 70, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"],
    )
    miss_by_age = (
        df_tmp.groupby("age_bin", observed=True)["MonthlyIncome_missing"]
        .mean()
        .mul(100)
    )
    bars = axes[1].bar(miss_by_age.index, miss_by_age.values,
                       color="steelblue", edgecolor="white")
    for bar, val in zip(bars, miss_by_age.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=9,
        )
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("MonthlyIncome Missing (%)")
    axes[1].set_title(
        "Missingness Rate by Age Group\n(observable driver: MAR evidence)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ── 2. Outlier & Sentinel Audit ───────────────────────────────────────────────

def plot_outlier_boxplots(df: pd.DataFrame) -> None:
    """
    Box plots for RevolvingUtilizationOfUnsecuredLines and DebtRatio
    to visualise the extreme-outlier problem before capping.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(y=df["RevolvingUtilizationOfUnsecuredLines"], ax=axes[0], color="#4C72B0")
    axes[0].set_title("RevolvingUtilizationOfUnsecuredLines", fontweight="bold")
    axes[0].set_ylabel("Value")
    sns.boxplot(y=df["DebtRatio"], ax=axes[1], color="#DD8452")
    axes[1].set_title("DebtRatio", fontweight="bold")
    axes[1].set_ylabel("Value")
    plt.suptitle("Box Plots — Outlier Detection", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier count for all continuous features.

    Returns a DataFrame with Q1, Q3, IQR, fence values, and outlier counts.
    """
    continuous_cols = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
    ]
    rows = []
    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "Feature": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "Lower Fence": lower, "Upper Fence": upper,
            "Outlier Count": n_out,
            "Outlier %": round(n_out / len(df) * 100, 2),
        })
    result = pd.DataFrame(rows).set_index("Feature")
    print("IQR-Based Outlier Summary:")
    return result


def sentinel_value_audit(df: pd.DataFrame) -> None:
    """
    Report counts of sentinel values (96, 98) in the three delinquency columns.
    """
    sentinel_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    for col in sentinel_cols:
        count_98 = (df[col] == 98).sum()
        count_96 = (df[col] == 96).sum()
        print(f"{col}:")
        print(f"  Value == 98: {count_98} rows ({count_98/len(df)*100:.3f}%)")
        print(f"  Value == 96: {count_96} rows")
        print(f"  Max value:   {df[col].max()}")
        print(f"  Value counts (top 5): {df[col].value_counts().head().to_dict()}")
        print()


def age_zero_check(df: pd.DataFrame) -> None:
    """Report rows with age == 0 (data entry errors)."""
    zero_age = (df["age"] == 0).sum()
    print(f"Rows with age == 0: {zero_age}")
    print(f"Age minimum: {df['age'].min()}")
    print("Age distribution (value counts for age < 18):")
    print(df[df["age"] < 18]["age"].value_counts().sort_index())


# ── 3. Class Imbalance ────────────────────────────────────────────────────────

def plot_class_imbalance(df: pd.DataFrame) -> None:
    """
    Print default rate statistics and plot target class counts + proportions.
    """
    target_counts = df["SeriousDlqin2yrs"].value_counts().sort_index()
    target_pct = df["SeriousDlqin2yrs"].value_counts(normalize=True).sort_index() * 100

    print("Target Class Counts:")
    print(pd.DataFrame({"Count": target_counts, "Percentage": target_pct.round(2)}))
    print(f"\nDefault rate: {target_pct[1]:.2f}%")
    print(f"Naive all-zero classifier accuracy: {target_pct[0]:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        ["Non-default (0)", "Default (1)"], target_counts.values,
        color=["#2196F3", "#F44336"], edgecolor="white", linewidth=0.8,
    )
    axes[0].set_title("Target Class Counts", fontweight="bold")
    axes[0].set_ylabel("Number of Borrowers")
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=11)

    axes[1].pie(
        target_counts.values,
        labels=["Non-default (0)", "Default (1)"],
        autopct="%1.2f%%", startangle=90,
        colors=["#2196F3", "#F44336"], explode=(0, 0.06),
    )
    axes[1].set_title("Target Class Proportions", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── 4. Univariate & Bivariate Distributions ───────────────────────────────────

ALL_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
    "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines", "NumberOfDependents",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def plot_univariate_distributions(df: pd.DataFrame,
                                  features: list = None) -> None:
    """
    Histogram + KDE for each feature (clipped at 99th percentile for clarity).
    """
    features = features or ALL_FEATURES
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    for i, col in enumerate(features):
        clip_val = df[col].quantile(0.99)
        data_clipped = df[col].clip(upper=clip_val).dropna()
        skew = df[col].skew()
        axes[i].hist(
            data_clipped, bins=50, color="steelblue", alpha=0.7,
            density=True, edgecolor="white", linewidth=0.3,
        )
        data_clipped.plot(kind="kde", ax=axes[i], color="darkblue", linewidth=1.5)
        axes[i].set_title(f"{col}\n(skew={skew:.2f})", fontsize=9, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Univariate Distributions (Histogram + KDE) — 99th percentile clipped",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.show()


def skewness_summary(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Return a skewness summary table for all features."""
    features = features or ALL_FEATURES
    skew_df = pd.DataFrame({
        "Skewness": df[features].skew(),
        "Right-skewed?": df[features].skew() > 1,
    }).sort_values("Skewness", ascending=False)
    print("Skewness Summary:")
    print(skew_df)
    return skew_df


def plot_correlation_heatmap(df: pd.DataFrame, features: list = None) -> None:
    """
    Pearson correlation heatmap (lower triangle only) for all features
    and the target.
    """
    features = features or ALL_FEATURES
    corr_matrix = df[features + ["SeriousDlqin2yrs"]].corr()
    plt.figure(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5,
        annot_kws={"size": 8}, vmin=-1, vmax=1,
    )
    plt.title("Pearson Correlation Heatmap (lower triangle)",
              fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def bivariate_default_rate(df: pd.DataFrame,
                            features: list = None) -> pd.DataFrame:
    """
    Split each feature at its median and compute the mean default rate in
    each half.  The difference in pp is a simple proxy for predictive strength.

    Returns a sorted DataFrame.
    """
    features = features or ALL_FEATURES
    rows = []
    for col in features:
        median_val = df[col].median()
        low_rate   = df[df[col] <= median_val]["SeriousDlqin2yrs"].mean()
        high_rate  = df[df[col] >  median_val]["SeriousDlqin2yrs"].mean()
        rows.append({
            "Feature": col,
            "Median": median_val,
            "Default Rate (≤ median)": round(low_rate * 100, 2),
            "Default Rate (> median)": round(high_rate * 100, 2),
            "Difference (pp)": round((high_rate - low_rate) * 100, 2),
        })
    result = (
        pd.DataFrame(rows)
        .set_index("Feature")
        .sort_values("Difference (pp)", ascending=False)
    )
    print("Default Rate by Feature Half (sorted by predictive strength):")
    return result


def plot_default_rate_by_decile(df: pd.DataFrame, features: list = None) -> None:
    """
    Bar charts showing mean default rate per decile bin for each feature.
    Non-linear relationships (J-curves, thresholds) are clearly visible.
    """
    features = features or ALL_FEATURES
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(features):
        temp = df[[col, "SeriousDlqin2yrs"]].dropna()
        temp = temp.copy()
        temp["bin"] = pd.qcut(temp[col], q=10, duplicates="drop")
        grouped = temp.groupby("bin", observed=True)["SeriousDlqin2yrs"].mean() * 100
        grouped.plot(kind="bar", ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col[:30], fontsize=8, fontweight="bold")
        axes[i].set_ylabel("Default Rate %")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", labelsize=6, rotation=45)

    plt.suptitle("Mean Default Rate by Feature Decile",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ── 5. Statistical Tests ──────────────────────────────────────────────────────

def mann_whitney_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U test for each continuous feature vs the binary target.
    Non-parametric — appropriate because features are not normally distributed.

    Returns a results DataFrame.
    """
    continuous_test_cols = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    alpha = 0.01
    results = []
    for col in continuous_test_cols:
        group0 = df[df["SeriousDlqin2yrs"] == 0][col].dropna()
        group1 = df[df["SeriousDlqin2yrs"] == 1][col].dropna()
        stat, p = stats.mannwhitneyu(group0, group1, alternative="two-sided")
        results.append({
            "Feature": col,
            "U-Statistic": round(stat, 2),
            "p-value": f"{p:.2e}",
            f"Significant (α={alpha})": "YES" if p < alpha else "NO",
        })
    result_df = pd.DataFrame(results).set_index("Feature")
    print("Mann-Whitney U Test Results (continuous features vs SeriousDlqin2yrs):")
    return result_df


def chi_squared_dependents(df: pd.DataFrame) -> None:
    """
    Chi-squared test for NumberOfDependents vs the target.
    Dependents is an ordinal count variable treated as categorical here.
    """
    alpha = 0.01
    temp = df[["NumberOfDependents", "SeriousDlqin2yrs"]].dropna()
    contingency = pd.crosstab(temp["NumberOfDependents"], temp["SeriousDlqin2yrs"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)

    print("Chi-Squared Test: NumberOfDependents vs SeriousDlqin2yrs")
    print(f"  Chi2 statistic    : {chi2:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value           : {p:.4e}")
    print(f"  Significant at α={alpha}: {'YES' if p < alpha else 'NO'}")
    print("\nContingency Table (crosstab):")
    print(contingency)
