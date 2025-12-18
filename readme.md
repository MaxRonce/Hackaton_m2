# Medical Mortality Prediction Pipeline (Hackathon M2)

Results by Maxime RONCERAY 

## Key Features
- **Gated Mixture of Experts (MoE)**: Specialized routing between Subjective (Questionnaire) and Objective (Lab/Exam) data components.
- **Gradient Boosting Machine (GBM)**: Standard GBM model for comparison.
- **Random Forest (RF)**: Standard RF model for comparison.
- **Logistic Regression (LR)**: Standard LR model for comparison.
- **Auto-Optimized Feature Selection**: Optuna-driven hyperparameter optimization and feature selection.
- **Advanced EDA**: Automated sparsity analysis with detailed justification for feature removal.
- **Interpretable SHAP**: Explainable AI features importance.


## Setup
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Exploratory Data Analysis (EDA)
Generate sparsity reports, target-conditional distributions, and filtering justifications:
```powershell
python ml_pipeline/main.py analyze --data-path data/data.csv --metadata-path data/features_metadata.csv
```
*Outputs: `outputs/eda/` including `potential_drops_details.csv`.*

### 2. Hyperparameter & FS Optimization
Find the best model parameters **and** feature selection thresholds simultaneously:
```powershell
# Optimize the "fancy" MoE model
python ml_pipeline/main.py optimize --model moe --n-trials 50

# Or optimize a standard GBM
python ml_pipeline/main.py optimize --model gbm --n-trials 50
```
*Best parameters are saved to `outputs/optimization/best_params_[model].json`.*

### 3. Training
Train with cross-validation and generate a detailed feature selection report:
```powershell
# Train MoE with optimized parameters
python ml_pipeline/main.py train --model moe --params-path outputs/optimization/best_params_moe.json
```

### 4. Prediction & Explainability
Generate submission files and SHAP waterfall plots with descriptive feature names:
```powershell
# Generate predictions
python ml_pipeline/main.py predict --model moe

# Generate SHAP explanations (Descriptive names are used automatically)
python ml_pipeline/main.py explain --model moe --data-path data/data.csv
```

### 5. Streamlit app

```powershell
streamlit run app.py
```


##  Project Structure
- `ml_pipeline/models/moe.py`: Gated Mixture of Experts implementation.
- `ml_pipeline/feature_filtering.py`: Logic for automated feature selection and reporting.
- `outputs/eda/`: Visual justifications for data filtering.
- `outputs/optimization/`: Best hyperparameter configurations.
- `outputs/feature_selection_report.json`: Justification for every dropped feature.
