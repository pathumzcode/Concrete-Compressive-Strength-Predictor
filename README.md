# CS-Predictor — Concrete Compressive Strength Prediction

[![CI](https://github.com/pathumzcode/CS-Predictor-An-ML-Based-Concrete-Compressive-Strength-Prediction-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/pathumzcode/CS-Predictor-An-ML-Based-Concrete-Compressive-Strength-Prediction-Model/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An ML-based regression model that predicts the **compressive strength of concrete** (MPa) using its mix composition and curing age.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Concrete compressive strength is a critical property in civil engineering. This project builds, evaluates, and compares several ML regression models to accurately predict it from the following input features:

| Feature | Unit | Description |
|---|---|---|
| Cement | kg/m³ | Cement component |
| Blast Furnace Slag | kg/m³ | Slag component |
| Fly Ash | kg/m³ | Fly ash component |
| Water | kg/m³ | Water content |
| Superplasticizer | kg/m³ | Superplasticizer component |
| Coarse Aggregate | kg/m³ | Coarse aggregate |
| Fine Aggregate | kg/m³ | Fine aggregate |
| Age | days | Curing age |
| **Concrete CS** | **MPa** | **Target: compressive strength** |

---

## Dataset

- **Source**: UCI Machine Learning Repository — [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- **Samples**: 1,030 instances
- **Features**: 8 input features + 1 target
- **File**: `data/raw/concrete_data.csv`

---

## Project Structure

```
CS-Predictor/
├── .github/workflows/      # CI/CD pipelines
├── data/
│   ├── raw/                # Original dataset
│   ├── processed/          # Cleaned & preprocessed data
│   └── external/           # External reference data
├── models/
│   ├── trained/            # Serialised model files (.joblib)
│   ├── preprocessing/      # Scaler / encoder files
│   └── metadata/           # Model metrics & metadata JSON
├── notebooks/              # Exploratory & modelling notebooks
├── src/                    # Source package
│   ├── data/               # Data loading & preprocessing
│   ├── features/           # Feature engineering & selection
│   ├── models/             # Training, prediction, selection
│   ├── evaluation/         # Metrics & evaluation helpers
│   ├── visualization/      # Plotting utilities
│   └── pipeline/           # End-to-end training pipeline
├── app/                    # Streamlit prediction app
├── scripts/                # CLI entry-point scripts
├── tests/                  # Unit & integration tests
├── config/                 # YAML configuration files
├── reports/                # Figures, CSVs, HTML reports
├── requirements.txt
└── pyproject.toml
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/pathumzcode/CS-Predictor-An-ML-Based-Concrete-Compressive-Strength-Prediction-Model.git
cd CS-Predictor-An-ML-Based-Concrete-Compressive-Strength-Prediction-Model

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
```

---

## Usage

### Train the Model

```bash
python scripts/train_model.py
```

### Evaluate the Model

```bash
python scripts/evaluate_model.py
```

### Generate Reports

```bash
python scripts/generate_reports.py
```

### Launch the Streamlit App

```bash
streamlit run app/app.py
```

### Run Tests

```bash
pytest --cov=src
```

---

## Model Performance

| Model | R² | RMSE (MPa) | MAE (MPa) |
|---|---|---|---|
| Random Forest | ~0.92 | ~4.5 | ~3.1 |
| XGBoost | ~0.93 | ~4.3 | ~2.9 |
| Gradient Boosting | ~0.91 | ~4.7 | ~3.3 |

> Results will be updated after full model training. See `reports/evaluation/` for detailed outputs.

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request into `main`

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
