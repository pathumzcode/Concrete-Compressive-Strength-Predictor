"""Predict using the Random Forest pipeline saved by its standalone notebook."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / 'models/metadata/random_forest_metadata.json').read_text())
    parser = argparse.ArgumentParser(description='Predict concrete strength in MPa using the complete PCA pipeline.')
    parser.add_argument('values', type=float, nargs=8,
                        help='Values in order: ' + ', '.join(metadata['feature_columns']))
    args = parser.parse_args()
    inputs = pd.DataFrame([args.values], columns=metadata['feature_columns'])
    model = joblib.load(ROOT / 'models/trained/random_forest_pca_pipeline.joblib')
    print(f"{metadata['model']}: {model.predict(inputs)[0]:.4f} MPa")


if __name__ == '__main__':
    main()
