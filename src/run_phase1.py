from config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    FIGURES_DIR,
)

from data_ingestion import load_data

from data_validation import (
    validate_dataset,
    print_validation_report
)

from eda import run_eda
def main():

    print("\n===================================")
    print("HOUSE PRICE PROJECT - PHASE 1")
    print("===================================")


    # ----------------------------------
    # 1. Load dataset
    # ----------------------------------

    df = load_data(RAW_DATA_PATH)


    # ----------------------------------
    # 2. Validate dataset
    # ----------------------------------

    validation_report = validate_dataset(
        df=df,
        target_column=TARGET_COLUMN
    )

    print_validation_report(
        validation_report
    )


    # ----------------------------------
    # 3. Run EDA
    # ----------------------------------

    run_eda(
        df=df,
        target_column=TARGET_COLUMN,
        figures_dir=FIGURES_DIR
    )


    print("\n===================================")
    print("PHASE 1 COMPLETED")
    print("===================================")


if __name__ == "__main__":

    main()