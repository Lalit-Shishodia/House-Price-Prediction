from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_house_price_model.pkl"
)


FEATURE_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]


def load_model():
    """
    Load the trained model pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run Phase 3 first."
        )

    return joblib.load(MODEL_PATH)


def validate_input(input_data: Dict[str, Any]) -> None:
    """
    Validate user input before prediction.
    """

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in input_data
    ]

    if missing_columns:
        raise ValueError(
            f"Missing input columns: {missing_columns}"
        )

    if input_data["area"] <= 0:
        raise ValueError(
            "Area must be greater than zero."
        )

    if input_data["bedrooms"] < 0:
        raise ValueError(
            "Bedrooms cannot be negative."
        )

    if input_data["bathrooms"] < 0:
        raise ValueError(
            "Bathrooms cannot be negative."
        )

    if input_data["stories"] < 0:
        raise ValueError(
            "Stories cannot be negative."
        )

    if input_data["parking"] < 0:
        raise ValueError(
            "Parking cannot be negative."
        )

    binary_columns = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]

    for column in binary_columns:
        if input_data[column] not in ["yes", "no"]:
            raise ValueError(
                f"{column} must be either 'yes' or 'no'."
            )

    allowed_furnishing = [
        "furnished",
        "semi-furnished",
        "unfurnished",
    ]

    if (
        input_data["furnishingstatus"]
        not in allowed_furnishing
    ):
        raise ValueError(
            "Invalid furnishing status."
        )


def predict_price(
    input_data: Dict[str, Any],
) -> float:
    """
    Predict house price from user input.
    """

    validate_input(input_data)

    model = load_model()

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_df)

    return float(prediction[0])