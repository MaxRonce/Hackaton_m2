import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to python path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.config import (
    DATA_DIR, MODELS_DIR, EDA_DIR, OUTPUTS_DIR, 
    TARGET_COLUMN, SEED
)
from ml_pipeline.utils.logger import setup_logger
from ml_pipeline.utils.io import save_pickle, load_pickle, save_json, load_json
from ml_pipeline.data.loader import load_data
from ml_pipeline.data.analysis import AutoEDA
from ml_pipeline.data.preprocessing import PreprocessingPipeline
from ml_pipeline.models.baseline import LogisticRegressionModel
from ml_pipeline.models.random_forest import RandomForestModel
from ml_pipeline.models.gradient_boosting import GradientBoostingModel
from ml_pipeline.evaluation.metrics import evaluate_model
from ml_pipeline.evaluation.plots import plot_roc_curve, plot_confusion_matrix, plot_metric_comparison
from ml_pipeline.optimization.optuna_search import HyperparameterOptimizer
from ml_pipeline.inference.predict import run_inference
from ml_pipeline.evaluation.plots import plot_roc_curve, plot_confusion_matrix, plot_metric_comparison
from ml_pipeline.optimization.optuna_search import HyperparameterOptimizer
from ml_pipeline.inference.predict import run_inference
from ml_pipeline.explanation.shap_explainer import run_shap_analysis
from ml_pipeline.utils.cache import load_from_cache, save_to_cache
from ml_pipeline.models.ensemble import load_models_from_paths

logger = setup_logger("main")

def get_data_splits(data_path):
    """Helper to get data splits with caching."""
    # Try cache
    cached_data = load_from_cache(data_path, "splits")
    if cached_data:
        # Ensure pipeline is saved to disk even if loaded from cache (e.g. if artifacts were cleaned)
        pipeline = cached_data[4]
        pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"
        if not pipeline_path.exists():
            save_pickle(pipeline, pipeline_path)
        return cached_data
        
    df = load_data(data_path)
    y = df[TARGET_COLUMN]
    
    from sklearn.model_selection import train_test_split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=SEED, stratify=y)
    
    pipeline = PreprocessingPipeline(target_col=TARGET_COLUMN)
    pipeline.fit(X_train_raw, y_train)
    
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    
    # Save pipeline too as it is fitted on this split
    save_pickle(pipeline, MODELS_DIR / "preprocessing_pipeline.pkl")
    
    data = (X_train, X_val, y_train, y_val, pipeline)
    save_to_cache(data, data_path, "splits")
    return data

def analyze(args):
    """Run EDA."""
    data_path = args.data_path
    df = load_data(data_path)
    eda = AutoEDA(df)
    eda.run()

def train(args):
    """Train a model."""
    logger.info(f"Training {args.model} model...")
    
    # Load Data with Cache
    X_train, X_val, y_train, y_val, pipeline = get_data_splits(args.data_path)
    
    # Preprocessing pipeline is already fitted and saved in get_data_splits if not cached, 
    # or just loaded if cached. We rely on get_data_splits to handle saving pipeline.
    
    # Initialise Model
    params = {}
    if args.params_path:
        logger.info(f"Loading parameters from {args.params_path}")
        params = load_json(args.params_path)

    model = None
    if args.model == "baseline":
        model = LogisticRegressionModel(**params)
    elif args.model == "rf":
        model = RandomForestModel(**params)
    elif args.model == "gbm":
        model = GradientBoostingModel(backend=args.gbm_backend, **params)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Fit
    model.fit(X_train, y_train.values)
    
    # Predict
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluate
    metrics = evaluate_model(y_val, y_pred, y_proba)
    logger.info(f"Metrics: {metrics}")
    
    # Save artifacts
    save_json(metrics, OUTPUTS_DIR / f"{args.model}_metrics.json")
    plot_roc_curve(y_val, y_proba, OUTPUTS_DIR / f"{args.model}_roc.png")
    plot_confusion_matrix(y_val, y_pred, OUTPUTS_DIR / f"{args.model}_cm.png")
    
    # Save Model
    model.save(MODELS_DIR / f"{args.model}_model.pkl")
    logger.info("Training complete.")

def optimize(args):
    """Hyperparameter Optimization."""
    
    X_train, X_val, y_train, y_val, pipeline = get_data_splits(args.data_path)
    
    models_to_opt = []
    if args.model == "all":
        models_to_opt = ["baseline", "rf", "gbm"]
    else:
        models_to_opt = [args.model]
        
    logger.info(f"Optimizing {models_to_opt} with {args.n_trials} trials... Backend for GBM: {args.gbm_backend}")

    # Pass gbm_backend to optimizer
    optimizer = HyperparameterOptimizer(models_to_opt, X_train, y_train.values, n_trials=args.n_trials, gbm_backend=args.gbm_backend)
    results = optimizer.optimize_all(MODELS_DIR)
    
    for model_name, params in results.items():
        save_json(params, MODELS_DIR / f"{model_name}_best_params.json")
        
    logger.info("Optimization complete details saved.")

def explain(args):
    """Explain model predictions using SHAP."""
    logger.info(f"Running Explanation for {args.model}...")
    
    # Load assets
    model_path = MODELS_DIR / f"{args.model}_model.pkl"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"
    
    model = load_pickle(model_path)
    pipeline = load_pickle(pipeline_path)
    
    # Load Data
    df = load_data(args.data_path)
    y = df[TARGET_COLUMN]
    
    # Split to act as background/test
    from sklearn.model_selection import train_test_split
    # We need a background set (train) and test set. 
    # Ideally should match train split, but for explanation we can just take a sample if needed.
    X_train_raw, X_test_raw, _, _ = train_test_split(df, y, test_size=0.2, random_state=SEED, stratify=y)
    
    X_train = pipeline.transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)
    
    # Get feature names if possible
    feature_names = None
    if hasattr(pipeline.pipeline, "get_feature_names_out"):
        feature_names = pipeline.pipeline.get_feature_names_out()
    
    # Run SHAP
    run_shap_analysis(model, X_train, X_test, OUTPUTS_DIR / "explanation", feature_names=feature_names)


def predict(args):
    """Inference."""
    run_inference(
        args.test_path, 
        args.model_path, 
        args.pipeline_path, 
        args.output_path
    )

def ensemble(args):
    """Run Soft Voting Ensemble."""
    logger.info("Running Ensemble Prediction...")
    
    ensemble_model = load_models_from_paths(args.model_paths)
    pipeline = load_pickle(args.pipeline_path)
    
    # Load and Preprocess Test Data
    df_test = load_data(args.test_path)
    X_test = pipeline.transform(df_test)
    
    # Predict
    # ensemble predict_proba returns (N,) array of probs for positive class
    preds_proba = ensemble_model.predict_proba(X_test)
    
    # Save
    ids = df_test["passengerid"] # Ensure consistency with predict.py, simplified here
    submission = pd.DataFrame({
        "id": ids,
        "prediction": np.round(preds_proba, 4)
    })
    
    submission.to_csv(args.output_path, index=False)
    logger.info(f"Ensemble predictions saved to {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze
    p_analyze = subparsers.add_parser("analyze", help="Run Automated EDA")
    p_analyze.add_argument("--data-path", type=str, required=True, help="Path to training data CSV")

    # Train
    p_train = subparsers.add_parser("train", help="Train a model")
    p_train.add_argument("--data-path", type=str, required=True, help="Path to training data CSV")
    p_train.add_argument("--model", type=str, choices=["baseline", "rf", "gbm"], required=True, help="Model to train")
    p_train.add_argument("--gbm-backend", type=str, choices=["lightgbm", "xgboost"], default="lightgbm", help="Backend for GBM model")
    p_train.add_argument("--params-path", type=str, help="Path to JSON file with optimized hyperparameters")

    # Optimize
    p_optimize = subparsers.add_parser("optimize", help="Run Hyperparameter Optimization")
    p_optimize.add_argument("--data-path", type=str, required=True, help="Path to training data CSV")
    p_optimize.add_argument("--model", type=str, choices=["baseline", "rf", "gbm", "all"], required=True, help="Model to optimize")
    p_optimize.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    p_optimize.add_argument("--gbm-backend", type=str, choices=["lightgbm", "xgboost"], default="lightgbm", help="Backend for GBM model")

    # Predict
    p_predict = subparsers.add_parser("predict", help="Run Inference")
    p_predict.add_argument("--test-path", type=str, required=True, help="Path to test data CSV")
    p_predict.add_argument("--model-path", type=str, required=True, help="Path to saved model pkl")
    p_predict.add_argument("--pipeline-path", type=str, default=str(MODELS_DIR / "preprocessing_pipeline.pkl"), help="Path to saved pipeline pkl")
    p_predict.add_argument("--output-path", type=str, default=str(OUTPUTS_DIR / "predictions.csv"), help="Path to save predictions")

    # Explain
    p_explain = subparsers.add_parser("explain", help="Run SHAP explanation")
    p_explain.add_argument("--data-path", type=str, required=True, help="Path to data CSV (for background/test)")
    p_explain.add_argument("--model", type=str, choices=["baseline", "rf", "gbm"], required=True, help="Model to explain (must be trained first)")

    # Ensemble
    p_ensemble = subparsers.add_parser("ensemble", help="Run Ensemble Prediction")
    p_ensemble.add_argument("--test-path", type=str, required=True, help="Path to test data CSV")
    p_ensemble.add_argument("--model-paths", type=str, nargs='+', required=True, help="List of paths to saved model pkls")
    p_ensemble.add_argument("--pipeline-path", type=str, default=str(MODELS_DIR / "preprocessing_pipeline.pkl"), help="Path to saved pipeline pkl")
    p_ensemble.add_argument("--output-path", type=str, default=str(OUTPUTS_DIR / "ensemble_predictions.csv"), help="Path to save predictions")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze(args)
    elif args.command == "train":
        train(args)
    elif args.command == "optimize":
        optimize(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "explain":
        explain(args)
    elif args.command == "ensemble":
        ensemble(args)

if __name__ == "__main__":
    main()
