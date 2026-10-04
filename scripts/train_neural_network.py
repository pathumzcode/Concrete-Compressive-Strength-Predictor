from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]
TARGET_COLUMN = "compressive_strength"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "concrete_data_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "neural_network_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing" / "neural_network_scaler.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata" / "neural_network_metadata.json"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def train_model():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=3000,
    )
    baseline_model.fit(X_train_scaled, y_train)
    baseline_r2 = r2_score(y_test, baseline_model.predict(X_test_scaled))

    search = GridSearchCV(
        estimator=MLPRegressor(
            activation="relu", solver="adam", random_state=42, max_iter=3000
        ),
        param_grid={
            "hidden_layer_sizes": [(64, 32), (128, 64)],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.0005, 0.001],
        },
        scoring="r2",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train_scaled, y_train)
    model = search.best_estimator_
    predictions = model.predict(X_test_scaled)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    print(f"R²: {r2:.4f}")
    print(f"Before tuning R2: {baseline_r2:.4f}")
    print(f"After tuning R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Best parameters: {search.best_params_}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_name": "MLPRegressor (Neural Network Regressor)",
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "train_test_split": {"test_size": 0.2, "random_state": 42},
        "key_hyperparameters": {
            "activation": "relu",
            "solver": "adam",
            "max_iter": 3000,
            "cv_folds": 3,
        },
        "best_parameters_used": {
            key: list(value) if key == "hidden_layer_sizes" else value
            for key, value in search.best_params_.items()
        },
        "before_tuning_test_r2": float(baseline_r2),
        "after_tuning_test_r2": float(r2),
        "test_metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    train_model()
