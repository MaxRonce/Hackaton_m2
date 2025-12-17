# Hackaton Machine Learning Pipeline

A generic, production-ready machine learning pipeline for binary classification on tabular data. Built with Python, Scikit-Learn, LightGBM, and Optuna.

## Features

- **Automated EDA**: Generate distribution plots and correlation maps automatically.
- **Modular Pipeline**: Clean separation of data, models, and evaluation.
- **Multiple Models**: Logistic Regression (Baseline), Random Forest, and Gradient Boosting (LightGBM/XGBoost).
- **Hyperparameter Tuning**: Bayesian optimization using Optuna.
- **Reproducibility**: Seeded random states and reusable preprocessing pipelines.
- **CLI Interface**: Full control via command line arguments.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The pipeline is controlled via `ml_pipeline/main.py`.

### 1. Data Analysis (EDA)
Generate automated reports (saved to `outputs/eda/`):
```bash
python ml_pipeline/main.py analyze --data-path data/train.csv
```

### 2. Model Training
Train a model. Available models: `baseline`, `rf` (Random Forest), `gbm` (Gradient Boosting).
```bash
# Default
python ml_pipeline/main.py train --model gbm --data-path data/train.csv

# With Optimized Params
python ml_pipeline/main.py train --model gbm --params-path outputs/models/gbm_best_params.json --data-path data/train.csv
```
Artifacts created:
- Model: `outputs/models/{model}_model.pkl`
- Metrics: `outputs/{model}_metrics.json`
- Plots: `outputs/{model}_roc.png`, `outputs/{model}_cm.png`

### 3. Hyperparameter Optimization
Optimize a model using Optuna. You can optimize a specific model or **all** models at once.
```bash
# Optimize a single model
python ml_pipeline/main.py optimize --model gbm --gbm-backend xgboost --data-path data/train.csv --n-trials 50

# Optimize ALL models (Baseline, RF, GBM) and compare results
python ml_pipeline/main.py optimize --model all --data-path data/train.csv --n-trials 20
```
Best search parameters are saved to `outputs/models/{model}_best_params.json` and a comparison plot is saved to `outputs/models/optimization_comparison.png`.

### 4. Explainability (SHAP)
Generate SHAP plots (Summary and Importance Bar) to explain model predictions.
```bash
python ml_pipeline/main.py explain --model gbm --data-path data/train.csv
```
Plots are saved to `outputs/explanation/`.

### 5. Ensemble Learning
Combine predictions from multiple trained models (Soft Voting).
```bash
python ml_pipeline/main.py ensemble \
    --test-path data/test.csv \
    --model-paths outputs/models/gbm_model.pkl outputs/models/rf_model.pkl \
    --output-path outputs/ensemble_submission.csv
```

### 6. Interactive Interface (Streamlit)
Launch the graphical interface for EDA and real-time inference.
```bash
streamlit run app.py
```

### 7. Inference
Generate predictions on a new dataset using a trained model.
```bash
python ml_pipeline/main.py predict \
    --test-path data/test.csv \
    --model-path outputs/models/gbm_model.pkl \
    --output-path outputs/predictions.csv
```
Output format is strictly `id,prediction`.

## Project Structure

```
ml_pipeline/
├── data/           # Data loading & preprocessing
├── models/         # Model implementations (RF, GBM, Baseline)
├── evaluation/     # Metrics & Plotting
├── optimization/   # Optuna Hyperparameter Search
├── inference/      # Prediction logic
├── utils/          # Logging & IO helpers
├── config.py       # Global Configuration
└── main.py         # CLI Entry Point
```
