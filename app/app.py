from pathlib import Path
import sys

import joblib
import pandas as pd

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "neural_network_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing" / "neural_network_scaler.joblib"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_strength(model, scaler, values):
    if len(values) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} feature values, got {len(values)}."
        )

    df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return float(prediction)


def run_prompted():
    print("Enter concrete mix values:")
    values = []
    for feature in FEATURE_COLUMNS:
        raw_value = input(f"{feature}: ")
        values.append(float(raw_value))

    model, scaler = load_artifacts()
    prediction = predict_strength(model, scaler, values)
    print(f"Predicted compressive strength: {prediction:.2f} MPa")


def run_with_args(raw_args):
    if len(raw_args) != len(FEATURE_COLUMNS):
        raise ValueError(
            "Provide exactly 8 numeric values in this order:\n"
            + ", ".join(FEATURE_COLUMNS)
        )

    values = [float(value) for value in raw_args]
    model, scaler = load_artifacts()
    prediction = predict_strength(model, scaler, values)
    print(f"Predicted compressive strength: {prediction:.2f} MPa")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            run_with_args(sys.argv[1:])
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)
    else:
        run_prompted()
