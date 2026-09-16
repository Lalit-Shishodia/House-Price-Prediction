import sys
from pathlib import Path

import pandas as pd


# Allow imports from the src directory
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODEL_RESULTS_PATH,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

from data_ingestion import load_data
from preprocessing import create_preprocessor
from train_models import train_and_compare_models


def main():
    print("\n========== PHASE 2 STARTED ==========")

    # ----------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------
    df = load_data(RAW_DATA_PATH)

    # ----------------------------------------------
    # 2. Confirm required columns
    # ----------------------------------------------
    required_columns = (
        [TARGET_COLUMN]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ----------------------------------------------
    # 3. Separate features and target
    # ----------------------------------------------
    X = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = df[TARGET_COLUMN].copy()

    # ----------------------------------------------
    # 4. Basic target validation
    # ----------------------------------------------
    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    if y.isnull().any():
        raise ValueError(
            "Target contains missing or non-numeric values."
        )

    if (y <= 0).any():
        raise ValueError(
            "Target must contain only positive values."
        )

    # ----------------------------------------------
    # 5. Create preprocessing pipeline
    # ----------------------------------------------
    preprocessor = create_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # ----------------------------------------------
    # 6. Train and compare models
    # ----------------------------------------------
    results_df, best_model = train_and_compare_models(
        X=X,
        y=y,
        preprocessor=preprocessor,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        results_path=MODEL_RESULTS_PATH,
        best_model_path=BEST_MODEL_PATH,
    )

    print("\n========== PHASE 2 COMPLETED ==========")

    print("\nSaved files:")
    print(f"- Results: {MODEL_RESULTS_PATH}")
    print(f"- Best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()