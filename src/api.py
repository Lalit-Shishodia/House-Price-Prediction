import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from predict import predict_price


app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices.",
    version="1.0.0",
)


class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    stories: int = Field(ge=0)
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    parking: int = Field(ge=0)
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal[
        "furnished",
        "semi-furnished",
        "unfurnished",
    ]


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(input_data: HouseInput):
    try:
        prediction = predict_price(
            input_data.model_dump()
        )

        return {
            "predicted_price": round(
                prediction,
                2,
            ),
            "currency": "dataset_currency",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from predict import predict_price


app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices.",
    version="1.0.0",
)


class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    stories: int = Field(ge=0)
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    parking: int = Field(ge=0)
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal[
        "furnished",
        "semi-furnished",
        "unfurnished",
    ]


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(input_data: HouseInput):
    try:
        prediction = predict_price(
            input_data.model_dump()
        )

        return {
            "predicted_price": round(
                prediction,
                2,
            ),
            "currency": "dataset_currency",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )