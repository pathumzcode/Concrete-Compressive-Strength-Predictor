import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "concrete_strength_model.pkl"

FEATURES = (
    {"name": "cement", "label": "Cement", "unit": "kg/m³"},
    {"name": "blast_furnace_slag", "label": "Blast furnace slag", "unit": "kg/m³"},
    {"name": "fly_ash", "label": "Fly ash", "unit": "kg/m³"},
    {"name": "water", "label": "Water", "unit": "kg/m³"},
    {"name": "superplasticizer", "label": "Superplasticizer", "unit": "kg/m³"},
    {"name": "coarse_aggregate", "label": "Coarse aggregate", "unit": "kg/m³"},
    {"name": "fine_aggregate", "label": "Fine aggregate", "unit": "kg/m³"},
    {"name": "age", "label": "Curing age", "unit": "days"},
)
FEATURE_COLUMNS = tuple(feature["name"] for feature in FEATURES)


def create_app(model=None):
    app = Flask(__name__)
    app.config["MODEL"] = model

    @app.route("/", methods=["GET", "POST"])
    def index():
        values = {}
        errors = {}
        prediction = None

        if request.method == "POST":
            for feature in FEATURES:
                name = feature["name"]
                raw_value = request.form.get(name, "").strip()
                values[name] = raw_value

                try:
                    value = float(raw_value)
                except ValueError:
                    errors[name] = "Enter a valid number."
                    continue

                if not math.isfinite(value):
                    errors[name] = "Enter a finite number."
                elif value < 0:
                    errors[name] = "Value cannot be negative."
                elif name == "age" and value == 0:
                    errors[name] = "Curing age must be greater than zero."
                else:
                    values[name] = value

            if not errors:
                model = app.config["MODEL"]
                if model is None:
                    model = joblib.load(MODEL_PATH)
                    app.config["MODEL"] = model

                inputs = pd.DataFrame(
                    [[values[name] for name in FEATURE_COLUMNS]],
                    columns=FEATURE_COLUMNS,
                )
                prediction = float(model.predict(inputs)[0])

        return (
            render_template(
                "index.html",
                features=FEATURES,
                values=values,
                errors=errors,
                prediction=prediction,
            ),
            400 if errors else 200,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
