from pathlib import Path
import pandas as pd


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load and perform basic structural checks on the CSV dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "Expected a CSV file."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    print(f"Dataset loaded successfully: {file_path}")
    print(f"Shape: {df.shape}")

    return df