from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 2),
    }


def create_actual_vs_predicted_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Save actual-vs-predicted scatter plot.
    """

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.7,
    )

    minimum = min(
        y_true.min(),
        y_pred.min(),
    )

    maximum = max(
        y_true.max(),
        y_pred.max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.title(
        "Actual vs Predicted House Prices"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "actual_vs_predicted.png",
        dpi=300,
    )

    plt.close()


def create_residual_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Save residual plot.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.title("Residual Analysis")

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "residual_plot.png",
        dpi=300,
    )

    plt.close()


def calculate_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
) -> pd.DataFrame:
    """
    Calculate feature importance using permutation importance.
    """

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance_Mean": result.importances_mean,
            "Importance_STD": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance_Mean",
        ascending=False,
    ).reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance_df,
        x="Importance_Mean",
        y="Feature",
    )

    plt.title(
        "Permutation Feature Importance"
    )

    plt.xlabel(
        "Importance "
        "(decrease in scoring performance)"
    )

    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "feature_importance.png",
        dpi=300,
    )

    plt.close()

    return importance_df