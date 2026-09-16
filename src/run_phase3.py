import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODEL_RESULTS_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)

from data_ingestion import load_data
from model_evaluation import (
    calculate_permutation_importance,
    calculate_regression_metrics,
    create_actual_vs_predicted_plot,
    create_residual_plot,
)
from model_tuning import (
    run_cross_validation,
    tune_gradient_boosting,
    tune_random_forest,
)
from preprocessing import create_preprocessor


def main():
    print("\n========== PHASE 3 STARTED ==========")

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # 1. Load data
    # ----------------------------------------------
    df = load_data(RAW_DATA_PATH)

    X = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if y.isnull().any():
        raise ValueError(
            "Target contains invalid values."
        )

    # ----------------------------------------------
    # 2. Train/test split
    # ----------------------------------------------
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    # ----------------------------------------------
    # 3. Create preprocessing object
    # ----------------------------------------------
    preprocessor = create_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # ----------------------------------------------
    # 4. Cross-validation
    # ----------------------------------------------
    print("\nRunning cross-validation...")

    cv_results = run_cross_validation(
        X=X_train,
        y=y_train,
        preprocessor=preprocessor,
        cv_splits=5,
    )

    cv_results_path = (
        REPORTS_DIR / "cross_validation_results.csv"
    )

    cv_results.to_csv(
        cv_results_path,
        index=False,
    )

    print("\nCross-validation results:")
    print(cv_results)

    # ----------------------------------------------
    # 5. Hyperparameter tuning
    # ----------------------------------------------
    print(
        "\nTuning Gradient Boosting..."
    )

    gradient_search = tune_gradient_boosting(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )

    print(
        "\nBest Gradient Boosting parameters:"
    )

    print(
        gradient_search.best_params_
    )

    print(
        "Best Gradient Boosting CV RMSE:",
        -gradient_search.best_score_,
    )

    print(
        "\nTuning Random Forest..."
    )

    random_forest_search = tune_random_forest(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )

    print(
        "\nBest Random Forest parameters:"
    )

    print(
        random_forest_search.best_params_
    )

    print(
        "Best Random Forest CV RMSE:",
        -random_forest_search.best_score_,
    )

    # ----------------------------------------------
    # 6. Compare tuned models on test data
    # ----------------------------------------------
    tuned_models = {
        "TunedGradientBoosting": (
            gradient_search.best_estimator_
        ),
        "TunedRandomForest": (
            random_forest_search.best_estimator_
        ),
    }

    tuning_results = []

    for model_name, model in tuned_models.items():
        predictions = model.predict(X_test)

        metrics = calculate_regression_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        metrics["Model"] = model_name

        tuning_results.append(metrics)

    tuning_results_df = pd.DataFrame(
        tuning_results
    )

    tuning_results_df = tuning_results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "MAPE",
        ]
    ]

    tuning_results_df = (
        tuning_results_df
        .sort_values(
            by="RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    tuning_results_path = (
        REPORTS_DIR / "tuning_results.csv"
    )

    tuning_results_df.to_csv(
        tuning_results_path,
        index=False,
    )

    print("\nTuned model results:")
    print(tuning_results_df)

    # ----------------------------------------------
    # 7. Select final model
    # ----------------------------------------------
    best_model_name = (
        tuning_results_df.iloc[0]["Model"]
    )

    final_model = tuned_models[
        best_model_name
    ]

    final_predictions = final_model.predict(
        X_test
    )

    final_metrics = calculate_regression_metrics(
        y_true=y_test,
        y_pred=final_predictions,
    )

    print("\n========== FINAL MODEL ==========")
    print("Model:", best_model_name)
    print("Metrics:", final_metrics)

    # ----------------------------------------------
    # 8. Create evaluation plots
    # ----------------------------------------------
    create_actual_vs_predicted_plot(
        y_true=y_test,
        y_pred=final_predictions,
        figures_dir=FIGURES_DIR,
    )

    create_residual_plot(
        y_true=y_test,
        y_pred=final_predictions,
        figures_dir=FIGURES_DIR,
    )

    # ----------------------------------------------
    # 9. Permutation feature importance
    # ----------------------------------------------
    importance_df = calculate_permutation_importance(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        figures_dir=FIGURES_DIR,
    )

    importance_path = (
        REPORTS_DIR / "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print("\nFeature importance:")
    print(importance_df)

    # ----------------------------------------------
    # 10. Save final model
    # ----------------------------------------------
    final_model_path = (
        MODELS_DIR / "final_house_price_model.pkl"
    )

    joblib.dump(
        final_model,
        final_model_path,
    )

    print(
        "\nFinal model saved to:",
        final_model_path,
    )

    print(
        "\n========== PHASE 3 COMPLETED =========="
    )


if __name__ == "__main__":
    main()