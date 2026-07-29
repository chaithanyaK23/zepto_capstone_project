"""Q2 Part A: Titanic profiling, cleaning, EDA charts, and written findings."""

from __future__ import annotations

import itertools
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODULE_DIR = Path(__file__).resolve().parent
CHART_DIR = MODULE_DIR / "charts"
RAW_CSV_PATH = MODULE_DIR / "titanic.csv"
CLEANED_CSV_PATH = MODULE_DIR / "titanic_cleaned.csv"
EDA_REPORT_PATH = MODULE_DIR / "eda_report.md"


def save_plot(path: Path) -> None:
    """Save the current matplotlib figure and close it to free memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def capture_dataframe_info(df: pd.DataFrame) -> str:
    """Capture df.info() text so it can be written into the report."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing-value counts and percentages for affected columns."""
    missing = df.isna().sum()
    report = (
        pd.DataFrame(
            {
                "missing_count": missing,
                "missing_percent": (missing / len(df) * 100).round(2),
            }
        )
        .query("missing_count > 0")
        .sort_values("missing_percent", ascending=False)
    )
    return report


def clean_titanic_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Apply the assignment's missing-value threshold rule."""
    clean_df = df.copy()
    missing_report = missing_value_report(clean_df)
    decisions: list[str] = []

    # Under 5% missing: drop those rows because very few records are affected.
    low_missing_columns = [
        column
        for column, row in missing_report.iterrows()
        if row["missing_percent"] < 5
    ]
    if low_missing_columns:
        before_rows = len(clean_df)
        clean_df = clean_df.dropna(subset=low_missing_columns).copy()
        dropped_rows = before_rows - len(clean_df)
        decisions.append(
            f"Dropped {dropped_rows} rows for {', '.join(low_missing_columns)} "
            "because each had under 5% missing values."
        )

    # 5%-30% missing: impute age with the median because it is numeric and moderately missing.
    if "age" in clean_df.columns and clean_df["age"].isna().any():
        age_missing_percent = df["age"].isna().mean() * 100
        age_median = clean_df["age"].median()
        clean_df["age"] = clean_df["age"].fillna(age_median)
        decisions.append(
            f"Imputed age with median {age_median:.1f} because it had "
            f"{age_missing_percent:.2f}% missing values, which is in the 5%-30% rule range."
        )

    # Very high missing: deck is too incomplete for reliable imputation, so encode missing explicitly.
    if "deck" in clean_df.columns:
        deck_missing_percent = df["deck"].isna().mean() * 100
        clean_df["deck"] = clean_df["deck"].astype("object").fillna("Missing")
        decisions.append(
            f"Encoded deck missing values as 'Missing' because deck had "
            f"{deck_missing_percent:.2f}% missing values, too high for reliable imputation."
        )

    # Keep categorical text columns simple for CSV output and later scripts.
    for column in clean_df.select_dtypes(include=["category"]).columns:
        clean_df[column] = clean_df[column].astype(str)

    return clean_df.reset_index(drop=True), decisions, missing_report


def iqr_outlier_count(series: pd.Series) -> tuple[int, float, float]:
    """Count outliers with the IQR rule and return count plus bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return count, lower, upper


def survival_rate(df: pd.DataFrame, mask: pd.Series) -> float:
    """Return survival rate for rows selected by a boolean mask."""
    return float(df.loc[mask, "survived"].mean() * 100)


def strongest_correlations(corr: pd.DataFrame) -> pd.DataFrame:
    """Rank off-diagonal correlation pairs by absolute value."""
    pairs = []
    for left, right in itertools.combinations(corr.columns, 2):
        value = corr.loc[left, right]
        pairs.append(
            {
                "feature_pair": f"{left} vs {right}",
                "correlation": round(float(value), 3),
                "absolute_correlation": round(abs(float(value)), 3),
            }
        )
    return pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False).head(2)


def create_univariate_charts(df: pd.DataFrame) -> list[str]:
    """Create histogram and box-plot charts for age and fare."""
    chart_paths: list[str] = []

    for column in ["age", "fare"]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f"{column.title()} Distribution")
        path = CHART_DIR / f"{column}_histogram.png"
        save_plot(path)
        chart_paths.append(path.relative_to(MODULE_DIR).as_posix())

        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[column])
        plt.title(f"{column.title()} Box Plot")
        path = CHART_DIR / f"{column}_boxplot.png"
        save_plot(path)
        chart_paths.append(path.relative_to(MODULE_DIR).as_posix())

    return chart_paths


def create_bivariate_and_story_charts(df: pd.DataFrame, corr: pd.DataFrame) -> dict[str, str]:
    """Create required heatmap and four multivariate story charts."""
    chart_paths: dict[str, str] = {}

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    path = CHART_DIR / "correlation_heatmap.png"
    save_plot(path)
    chart_paths["correlation_heatmap"] = path.relative_to(MODULE_DIR).as_posix()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None)
    plt.ylabel("Survival Rate")
    plt.title("Survival by Sex and Passenger Class")
    path = CHART_DIR / "story_survival_by_sex_class.png"
    save_plot(path)
    chart_paths["story_survival_by_sex_class"] = path.relative_to(MODULE_DIR).as_posix()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="survived", y="age", hue="sex")
    plt.title("Age by Survival Outcome and Sex")
    path = CHART_DIR / "story_age_by_survival_sex.png"
    save_plot(path)
    chart_paths["story_age_by_survival_sex"] = path.relative_to(MODULE_DIR).as_posix()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survived")
    plt.title("Fare by Class and Survival Outcome")
    path = CHART_DIR / "story_fare_by_class_survival.png"
    save_plot(path)
    chart_paths["story_fare_by_class_survival"] = path.relative_to(MODULE_DIR).as_posix()

    story_df = df.copy()
    story_df["family_size"] = story_df["sibsp"] + story_df["parch"] + 1
    story_df["family_group"] = pd.cut(
        story_df["family_size"],
        bins=[0, 1, 4, 20],
        labels=["Alone", "Small family", "Large family"],
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=story_df, x="family_group", y="survived", errorbar=None)
    plt.ylabel("Survival Rate")
    plt.xlabel("Family Group")
    plt.title("Survival by Family Size")
    path = CHART_DIR / "story_family_size_survival.png"
    save_plot(path)
    chart_paths["story_family_size_survival"] = path.relative_to(MODULE_DIR).as_posix()

    return chart_paths


def create_standardization_check(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize age and fare for the EDA sanity check only."""
    summary_rows = []

    for column in ["age", "fare"]:
        z_values = (df[column] - df[column].mean()) / df[column].std(ddof=0)
        summary_rows.append(
            {
                "column": column,
                "before_mean": round(float(df[column].mean()), 4),
                "before_std": round(float(df[column].std(ddof=0)), 4),
                "after_mean": round(float(z_values.mean()), 4),
                "after_std": round(float(z_values.std(ddof=0)), 4),
            }
        )

        plt.figure(figsize=(8, 5))
        sns.kdeplot(df[column], label=f"Original {column}", fill=True, alpha=0.35)
        sns.kdeplot(z_values, label=f"Z-score {column}", fill=True, alpha=0.35)
        plt.legend()
        plt.title(f"{column.title()} Before and After Z-score Standardization")
        save_plot(CHART_DIR / f"{column}_standardization_check.png")

    return pd.DataFrame(summary_rows)


def write_eda_report(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    missing_report: pd.DataFrame,
    decisions: list[str],
    outlier_table: pd.DataFrame,
    fare_stats: dict[str, float],
    survival_tables: dict[str, pd.DataFrame],
    corr: pd.DataFrame,
    top_corr: pd.DataFrame,
    standardization_summary: pd.DataFrame,
    chart_paths: dict[str, str],
) -> None:
    """Write all EDA outputs and interpretations into a markdown report."""
    fare_shape = (
        "right-skewed"
        if fare_stats["mean"] > fare_stats["median"] >= fare_stats["mode"]
        else "left-skewed"
        if fare_stats["mean"] < fare_stats["median"] <= fare_stats["mode"]
        else "approximately symmetric"
    )

    lines = [
        "# Q2 Part A: EDA Report",
        "",
        "## Dataset Profile",
        "",
        f"- Raw shape: {raw_df.shape}",
        f"- Cleaned shape: {clean_df.shape}",
        "",
        "### df.info()",
        "",
        "```text",
        capture_dataframe_info(raw_df).strip(),
        "```",
        "",
        "### df.describe()",
        "",
        raw_df.describe(include="all").transpose().to_markdown(),
        "",
        "## Missing Values and Cleaning Decisions",
        "",
        missing_report.to_markdown(),
        "",
    ]

    for decision in decisions:
        lines.append(f"- {decision}")

    lines.extend(
        [
            "",
            "## Univariate Analysis",
            "",
            outlier_table.to_markdown(index=False),
            "",
            (
                f"Fare mean = {fare_stats['mean']:.2f}, median = {fare_stats['median']:.2f}, "
                f"and mode = {fare_stats['mode']:.2f}. Because the mean is greater than the "
                f"median and the median is greater than or equal to the mode, fare is {fare_shape}; "
                "a smaller number of very expensive tickets pulls the average upward."
            ),
            "",
            "## Bivariate Analysis",
            "",
            "### Survival by Sex",
            "",
            survival_tables["sex"].to_markdown(index=False),
            "",
            "### Survival by Passenger Class",
            "",
            survival_tables["pclass"].to_markdown(index=False),
            "",
            "### Survival by Sex and Passenger Class",
            "",
            survival_tables["sex_pclass"].to_markdown(index=False),
            "",
            "## Correlation Matrix",
            "",
            "The heatmap uses exactly survived, pclass, age, sibsp, parch, and fare. "
            "The derived boolean columns adult_male and alone are excluded.",
            "",
            corr.round(3).to_markdown(),
            "",
            "### Two Strongest Off-Diagonal Correlations",
            "",
            top_corr.to_markdown(index=False),
            "",
            (
                f"The strongest pair is {top_corr.iloc[0]['feature_pair']} with correlation "
                f"{top_corr.iloc[0]['correlation']}. The second strongest pair is "
                f"{top_corr.iloc[1]['feature_pair']} with correlation {top_corr.iloc[1]['correlation']}. "
                "These values summarize the two largest linear relationships among the selected numeric columns."
            ),
            "",
            "## Multivariate Data Story",
            "",
            f"![Survival by sex and class]({chart_paths['story_survival_by_sex_class']})",
            "",
            "Women show a much higher survival rate than men across passenger classes. First-class passengers also survive more often than lower-class passengers, suggesting that both gender and class shaped access to safety.",
            "",
            f"![Age by survival and sex]({chart_paths['story_age_by_survival_sex']})",
            "",
            "The age chart shows that survival patterns differ by sex across a wide age range. It also shows that age alone does not explain survival as strongly as sex and class do.",
            "",
            f"![Fare by class and survival]({chart_paths['story_fare_by_class_survival']})",
            "",
            "Higher fares are concentrated in first class, and first-class passengers have better survival outcomes. Fare is therefore partly a proxy for class and access to better locations or resources on the ship.",
            "",
            f"![Survival by family size]({chart_paths['story_family_size_survival']})",
            "",
            "Passengers traveling in small family groups tend to do better than passengers traveling alone or in larger groups. This suggests that moderate family support may have helped, while large groups may have been harder to coordinate during evacuation.",
            "",
            "## Exploratory Standardization Check",
            "",
            standardization_summary.to_markdown(index=False),
            "",
            "The standardized age and fare columns have means near 0 and standard deviations near 1. This check is only for EDA; the modeling pipeline performs its own train-only scaling later.",
            "",
        ]
    )

    EDA_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run Part A from raw data loading through cleaned CSV and EDA report."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Required Step 1: load Titanic exactly once from Seaborn and save the raw fallback.
    raw_df = sns.load_dataset("titanic")
    raw_df.to_csv(RAW_CSV_PATH, index=False)

    print("Loaded Titanic once with sns.load_dataset('titanic').")
    print(f"Raw shape: {raw_df.shape}")
    print(capture_dataframe_info(raw_df))
    print(raw_df.describe(include="all"))

    # Required Step 2: compute missing values and apply threshold-based cleaning.
    clean_df, decisions, missing_report = clean_titanic_data(raw_df)
    clean_df.to_csv(CLEANED_CSV_PATH, index=False)

    # Required Step 3: univariate histograms, box plots, outliers, and fare skewness.
    create_univariate_charts(clean_df)
    outlier_rows = []
    for column in ["age", "fare"]:
        count, lower, upper = iqr_outlier_count(clean_df[column])
        outlier_rows.append(
            {
                "column": column,
                "outlier_count": count,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            }
        )
    outlier_table = pd.DataFrame(outlier_rows)
    fare_stats = {
        "mean": float(clean_df["fare"].mean()),
        "median": float(clean_df["fare"].median()),
        "mode": float(clean_df["fare"].mode().iloc[0]),
    }

    # Required Step 4: boolean-mask survival breakdowns.
    survival_tables = {
        "sex": pd.DataFrame(
            [
                {"group": sex, "survival_rate_percent": round(survival_rate(clean_df, clean_df["sex"] == sex), 2)}
                for sex in sorted(clean_df["sex"].dropna().unique())
            ]
        ),
        "pclass": pd.DataFrame(
            [
                {
                    "group": f"pclass_{pclass}",
                    "survival_rate_percent": round(survival_rate(clean_df, clean_df["pclass"] == pclass), 2),
                }
                for pclass in sorted(clean_df["pclass"].dropna().unique())
            ]
        ),
    }
    sex_pclass_rows = []
    for sex in sorted(clean_df["sex"].dropna().unique()):
        for pclass in sorted(clean_df["pclass"].dropna().unique()):
            mask = (clean_df["sex"] == sex) & (clean_df["pclass"] == pclass)
            sex_pclass_rows.append(
                {
                    "sex": sex,
                    "pclass": pclass,
                    "survival_rate_percent": round(survival_rate(clean_df, mask), 2),
                }
            )
    survival_tables["sex_pclass"] = pd.DataFrame(sex_pclass_rows)

    # Required Step 5: exact six-column correlation matrix and heatmap.
    corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = clean_df[corr_columns].corr()
    top_corr = strongest_correlations(corr)

    # Required Step 6: multivariate charts and EDA-only standardization check.
    chart_paths = create_bivariate_and_story_charts(clean_df, corr)
    standardization_summary = create_standardization_check(clean_df)

    write_eda_report(
        raw_df,
        clean_df,
        missing_report,
        decisions,
        outlier_table,
        fare_stats,
        survival_tables,
        corr,
        top_corr,
        standardization_summary,
        chart_paths,
    )

    print(f"Saved raw fallback CSV: {RAW_CSV_PATH}")
    print(f"Saved cleaned CSV: {CLEANED_CSV_PATH}")
    print(f"Saved EDA report: {EDA_REPORT_PATH}")


if __name__ == "__main__":
    main()
