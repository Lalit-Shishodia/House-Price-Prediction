
import numpy as np
import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    target_column: str
) -> dict:
    """
    Validate a house price dataset.

    Returns a dictionary containing
    data quality results.
    """

    report = {}

    # --------------------------------------
    # 1. Dataset shape
    # --------------------------------------

    report["rows"] = int(df.shape[0])
    report["columns"] = int(df.shape[1])


    # --------------------------------------
    # 2. Column names
    # --------------------------------------

    report["column_names"] = df.columns.tolist()


    # --------------------------------------
    # 3. Duplicate records
    # --------------------------------------

    report["duplicate_rows"] = int(
        df.duplicated().sum()
    )


    # --------------------------------------
    # 4. Missing values
    # --------------------------------------

    missing_values = df.isnull().sum()

    report["missing_values"] = (
        missing_values[missing_values > 0]
        .to_dict()
    )


    # --------------------------------------
    # 5. Target validation
    # --------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "not found in dataset."
        )

    report["target_column"] = target_column

    report["target_dtype"] = str(
        df[target_column].dtype
    )


    # --------------------------------------
    # 6. Target numeric validation
    # --------------------------------------

    target_numeric = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    report["target_non_numeric_values"] = int(
        target_numeric.isnull().sum()
    )


    # --------------------------------------
    # 7. Invalid target values
    # --------------------------------------

    report["target_zero_or_negative"] = int(
        (target_numeric <= 0).sum()
    )


    # --------------------------------------
    # 8. Infinite values
    # --------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    report["infinite_values"] = int(
        np.isinf(numeric_df).sum().sum()
    )


    # --------------------------------------
    # 9. Data types
    # --------------------------------------

    report["numeric_columns"] = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    report["categorical_columns"] = (
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )


    # --------------------------------------
    # 10. Duplicate column names
    # --------------------------------------

    report["duplicate_column_names"] = (
        df.columns[
            df.columns.duplicated()
        ].tolist()
    )


    return report


def print_validation_report(report: dict):

    print("\n========== DATA VALIDATION REPORT ==========")

    print(
        f"Rows: {report['rows']}"
    )

    print(
        f"Columns: {report['columns']}"
    )

    print(
        f"Duplicate rows: "
        f"{report['duplicate_rows']}"
    )

    print(
        f"Missing values: "
        f"{report['missing_values']}"
    )

    print(
        f"Target column: "
        f"{report['target_column']}"
    )

    print(
        f"Target data type: "
        f"{report['target_dtype']}"
    )

    print(
        f"Non-numeric target values: "
        f"{report['target_non_numeric_values']}"
    )

    print(
        f"Zero or negative target values: "
        f"{report['target_zero_or_negative']}"
    )

    print(
        f"Infinite values: "
        f"{report['infinite_values']}"
    )

    print(
        f"Numeric columns: "
        f"{report['numeric_columns']}"
    )

    print(
        f"Categorical columns: "
        f"{report['categorical_columns']}"
    )

    print(
        f"Duplicate column names: "
        f"{report['duplicate_column_names']}"
    )