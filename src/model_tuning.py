from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """
    Evaluate baseline models using K-Fold cross-validation.
    """

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        results.append(
            {
                "Model": model_name,
                "CV_MAE_Mean": round(
                    -scores["test_mae"].mean(),
                    2,
                ),
                "CV_RMSE_Mean": round(
                    -scores["test_rmse"].mean(),
                    2,
                ),
                "CV_R2_Mean": round(
                    scores["test_r2"].mean(),
                    4,
                ),
                "CV_RMSE_STD": round(
                    scores["test_rmse"].std(),
                    2,
                ),
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="CV_RMSE_Mean",
        ascending=True,
    ).reset_index(drop=True)

    return results_df


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> GridSearchCV:
    """
    Tune Gradient Boosting using GridSearchCV.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=42,
                ),
            ),
        ]
    )

    parameter_grid = {
        "model__n_estimators": [
            100,
            200,
            300,
        ],
        "model__learning_rate": [
            0.03,
            0.05,
            0.1,
        ],
        "model__max_depth": [
            2,
            3,
            4,
        ],
        "model__min_samples_split": [
            2,
            5,
            10,
        ],
        "model__min_samples_leaf": [
            1,
            2,
            4,
        ],
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    return grid_search


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> RandomizedSearchCV:
    """
    Tune Random Forest using RandomizedSearchCV.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    parameter_distributions = {
        "model__n_estimators": [
            200,
            300,
            500,
        ],
        "model__max_depth": [
            None,
            5,
            10,
            15,
            20,
        ],
        "model__min_samples_split": [
            2,
            5,
            10,
        ],
        "model__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "model__max_features": [
            0.5,
            0.8,
            1.0,
        ],
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(
        X_train,
        y_train,
    )

    return random_search