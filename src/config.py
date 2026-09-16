from pathlib import Path


# --------------------------------------------------
# Project directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

RAW_DATA_PATH = RAW_DATA_DIR / "House_price.csv"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "processed_house_prices.csv"
)

MODEL_RESULTS_PATH = (
    REPORTS_DIR / "model_results.csv"
)

BEST_MODEL_PATH = (
    MODELS_DIR / "best_house_price_model.pkl"
)


# --------------------------------------------------
# Target and features
# --------------------------------------------------

TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]


# --------------------------------------------------
# Model-training settings
# --------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42