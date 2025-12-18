# Medical Mortality Prediction Pipeline (Hackathon M2)

This repository contains an adapted Machine Learning pipeline designed for a sparse medical dataset (~59k rows, ~1.5k features). It is optimized for F1-score and utilizes LightGBM/XGBoost with robust preprocessing for handling missing data.

## Features
- **Advanced EDA**: Automated sparsity and metadata-driven analysis.
- **Robust Preprocessing**: Handles high missingness (>95%) with specific indicators and categorical encoding.
- **Optimization**: Integrated Optuna support for hyperparameter tuning.

## Setup
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```


## Usage

### 1. Exploratory Data Analysis (EDA)
Generate sparsity reports and distribution plots:
```powershell
python ml_pipeline/main.py analyze --data-path data/data.csv --metadata-path data/features_metadata.csv
```
Outputs are saved to `outputs/eda/`.

### 2. Training
**Fast Debug Run (Subsample):**
```powershell
python ml_pipeline/main.py train --model baseline --data-path data/data.csv --subsample 1000
```

**Full Training (LightGBM):**
```powershell
python ml_pipeline/main.py train --model gbm --data-path data/data.csv
```

### 3. Hyperparameter Optimization
Find the best parameters using Optuna:
```powershell
python ml_pipeline/main.py optimize --model gbm --data-path data/data.csv --n-trials 50
```
Best parameters are saved to `outputs/optimization/best_params_gbm.json`.

**Train with Optimized Parameters:**
```powershell
python ml_pipeline/main.py train --model gbm --data-path data/data.csv --params-path outputs/optimization/best_params_gbm.json
```

### 4. Prediction (Inference)
Generate predictions for submission (defaults to predicting on **ALL** rows in `data.csv`):
```powershell
python ml_pipeline/main.py predict --model gbm
```
Output: `outputs/predictions.csv` (One column: `y`).

**Predict on specific Test Indexes:**
```powershell
python ml_pipeline/main.py predict --model gbm --test-indexes data/test_indexes.csv
```

## outputs structure
- `outputs/models/`: Saved models and pipelines.
- `outputs/eda/`: EDA plots.
- `outputs/optimization/`: Best hyperparameters.
- `outputs/predictions.csv`: Final inference file.
