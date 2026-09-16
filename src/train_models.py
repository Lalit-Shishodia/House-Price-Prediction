from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 2),
    }


def build_models(preprocessor) -> Dict[str, Pipeline]:
    """
    Create model pipelines.
    """

    models = {
        "LinearRegression": LinearRegression(),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=2,
            random_state=42,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    pipelines = {}

    for model_name, model in models.items():
        pipelines[model_name] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    return pipelines


def train_and_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    test_size: float,
    random_state: int,
    results_path: Path,
    best_model_path: Path,
) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Split data, train models, compare metrics, and save the best model.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    pipelines = build_models(preprocessor)

    results = []
    trained_models = {}

    for model_name, pipeline in pipelines.items():
        print(f"\nTraining: {model_name}")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        metrics["Model"] = model_name

        results.append(metrics)
        trained_models[model_name] = pipeline

        print("Metrics:")
        print(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "MAPE",
        ]
    ]

    # Lower RMSE is better
    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True,
    ).reset_index(drop=True)

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    joblib.dump(
        best_model,
        best_model_path,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df)

    print(
        f"\nBest model based on RMSE: "
        f"{best_model_name}"
    )

    print(
        f"Model saved to: "
        f"{best_model_path}"
    )

    return results_df, best_model