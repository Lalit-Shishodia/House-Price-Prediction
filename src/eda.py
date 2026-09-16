from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df: pd.DataFrame, target_column: str, figures_dir: Path) -> None:
    """
    Perform basic exploratory data analysis and save reports/figures.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== EDA STARTED ==========")

    # --------------------------------------------------
    # 1. Dataset overview
    # --------------------------------------------------
    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nStatistical Summary:")
    print(df.describe(include="all").T)

    # Save dataset overview
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    overview_path = figures_dir.parent / "eda_overview.json"

    with open(overview_path, "w", encoding="utf-8") as file:
        json.dump(overview, file, indent=4)

    # --------------------------------------------------
    # 2. Missing-value analysis
    # --------------------------------------------------
    missing_values = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    missing_values.columns = ["Column", "MissingValues"]

    missing_path = figures_dir.parent / "missing_values.csv"
    missing_values.to_csv(missing_path, index=False)

    print("\nMissing Values:")
    print(missing_values)

    # --------------------------------------------------
    # 3. Duplicate rows
    # --------------------------------------------------
    duplicate_count = int(df.duplicated().sum())

    print(f"\nDuplicate Rows: {duplicate_count}")

    # --------------------------------------------------
    # 4. Target-column analysis
    # --------------------------------------------------
    if target_column not in df.columns:
        print(
            f"\nWARNING: Target column '{target_column}' "
            "was not found. Skipping target analysis."
        )
    else:
        target = df[target_column]

        # Target histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=target.dropna().to_frame(name=target_column),
            x=target_column,
            kde=True,
        )
        plt.title(f"Distribution of {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        target_histogram_path = (
            figures_dir / "target_distribution.png"
        )

        plt.savefig(target_histogram_path, dpi=300)
        plt.close()

        # Target boxplot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=target.dropna())
        plt.title(f"Boxplot of {target_column}")
        plt.xlabel(target_column)
        plt.tight_layout()

        target_boxplot_path = (
            figures_dir / "target_boxplot.png"
        )

        plt.savefig(target_boxplot_path, dpi=300)
        plt.close()

    # --------------------------------------------------
    # 5. Numeric-column analysis
    # --------------------------------------------------
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    print("\nNumeric Columns:")
    print(numeric_columns)

    if numeric_columns:
        numeric_summary = df[numeric_columns].describe().T

        numeric_summary_path = (
            figures_dir.parent / "numeric_summary.csv"
        )

        numeric_summary.to_csv(numeric_summary_path)

    # --------------------------------------------------
    # 6. Correlation analysis
    # --------------------------------------------------
    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr()

        correlation_path = (
            figures_dir.parent / "correlation_matrix.csv"
        )

        correlation_matrix.to_csv(correlation_path)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )

        plt.title("Correlation Matrix")
        plt.tight_layout()

        heatmap_path = (
            figures_dir / "correlation_heatmap.png"
        )

        plt.savefig(heatmap_path, dpi=300)
        plt.close()

    # --------------------------------------------------
    # 7. Categorical-column analysis
    # --------------------------------------------------
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nCategorical Columns:")
    print(categorical_columns)

    categorical_summary = {}

    for column in categorical_columns:
        value_counts = (
            df[column]
            .value_counts(dropna=False)
            .head(20)
            .to_dict()
        )

        categorical_summary[column] = {
            str(key): int(value)
            for key, value in value_counts.items()
        }

    categorical_summary_path = (
        figures_dir.parent / "categorical_summary.json"
    )

    with open(
        categorical_summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            categorical_summary,
            file,
            indent=4,
            default=str,
        )

    print("\n========== EDA COMPLETED ==========")
    print(f"Reports saved in: {figures_dir.parent}")
    print(f"Figures saved in: {figures_dir}")


if __name__ == "__main__":
    print("EDA module loaded successfully.")