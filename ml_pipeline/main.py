import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to python path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.config import (
    DATA_DIR, MODELS_DIR, EDA_DIR, OUTPUTS_DIR, 
    TARGET_COLUMN, ID_COLUMN, SEED
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
    
    # Merge with ground truth if target missing (Optimization/Splits need target)
    if TARGET_COLUMN not in df.columns:
        ground_truth_path = Path(data_path).parent / "ground_truth_train.csv"
        if ground_truth_path.exists():
            labels = pd.read_csv(ground_truth_path)
            if ID_COLUMN in df.columns and ID_COLUMN in labels.columns:
                 df = df.merge(labels, on=ID_COLUMN, how="inner")
                 logger.info("Merged data with ground_truth_train.csv for splits.")
                 
    if TARGET_COLUMN not in df.columns:
         raise ValueError(f"Target column {TARGET_COLUMN} not found in {data_path} (even after attempted merge).")
         
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
    from ml_pipeline.config import EDA_DIR
    from ml_pipeline.eda_advanced import run_advanced_eda
    
    logger.info(f"Running analysis on {args.data_path}")
    run_advanced_eda(
        data_path=Path(args.data_path),
        metadata_path=Path(args.metadata_path),
        output_dir=EDA_DIR
    )


from sklearn.model_selection import StratifiedKFold
from ml_pipeline.feature_filtering import FeatureSelector
from ml_pipeline.optimization.threshold import optimize_threshold

def train(args):
    """Train a model with CV and Feature Selection."""
    logger.info(f"Training {args.model} model...")
    
    # 1. Load Data
    # Merge data and ground truth
    df = load_data(args.data_path)
    # We need labels. Assuming ground_truth_train.csv is in the data dir or we use the labels in data.csv if merged?
    # User said "Provided files: data.csv, ground_truth_train.csv". 
    # Usually data.csv is features, ground_truth is labels.
    # Let's check if 'y' (target) is in data.csv. If not, merge.
    # Earlier head of data.csv showed SEQN...
    # head of ground_truth showed SEQN, y.
    # So we must merge.
    
    ground_truth_path = Path(args.data_path).parent / "ground_truth_train.csv"
    if ground_truth_path.exists():
        labels = pd.read_csv(ground_truth_path)
        # Merge on ID
        if ID_COLUMN in df.columns and ID_COLUMN in labels.columns:
            df = df.merge(labels, on=ID_COLUMN, how="inner")
        else:
             logger.warning("Could not merge labels automatically. Assuming target is in data.")
    
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column {TARGET_COLUMN} not found in data.")
        
    # User Request: Save merged dataset
    merged_path = Path(args.data_path).parent / "data_merged.csv"
    if not merged_path.exists():
        df.to_csv(merged_path, index=False)
        logger.info(f"Saved merged dataset to {merged_path}")
        
    # User Request: Subsample
    if args.subsample:
        logger.warning(f"Subsampling data to {args.subsample} rows for debugging.")
        df = df.head(args.subsample)
        
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    
    # 2. Cross Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(df))
    oof_probs = np.zeros(len(df))
    
    fold_f1s = []
    
    logger.info("Starting 5-Fold Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # A. Feature Filtering
        # Create output path for report only for first fold or separate?
        # Let's not save report for every fold to avoid clutter, or save to distinct files.
        selector = FeatureSelector(output_report_path=None) 
        selector.fit(X_train_fold, y_train_fold)
        X_train_sel = selector.transform(X_train_fold)
        X_val_sel = selector.transform(X_val_fold)
        
        # B. Preprocessing
        pipeline = PreprocessingPipeline(target_col=TARGET_COLUMN)
        pipeline.fit(X_train_sel, y_train_fold)
        X_train_proc = pipeline.transform(X_train_sel)
        X_val_proc = pipeline.transform(X_val_sel)
        
        # C. Model
        params = {}
        if args.params_path:
             params = load_json(args.params_path)
             
        model = None
        if args.model == "baseline":
            model = LogisticRegressionModel(**params)
        elif args.model == "rf":
            model = RandomForestModel(**params)
        elif args.model == "gbm":
            model = GradientBoostingModel(backend=args.gbm_backend, **params)
            
        # Get categorical indices for LightGBM
        cat_indices = pipeline.get_categorical_indices()
        
        # Fit
        if args.model == "gbm" and args.gbm_backend == "lightgbm":
             model.fit(X_train_proc, y_train_fold.values, categorical_feature=cat_indices)
        else:
             model.fit(X_train_proc, y_train_fold.values)
             
        # Predict
        probs = model.predict_proba(X_val_proc)[:, 1]
        oof_probs[val_idx] = probs
        
        # We don't have optimal threshold yet, use 0.5 for fold metrics log
        preds = (probs >= 0.5).astype(int)
        fold_score = evaluate_model(y_val_fold, preds, probs)["f1"]
        fold_f1s.append(fold_score)
        
        logger.info(f"Fold {fold+1} F1: {fold_score:.4f}")
        
    logger.info(f"Mean CV F1 (thresh=0.5): {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    
    # 3. Optimization
    # Optimize threshold on OOF
    best_thresh, best_f1 = optimize_threshold(y, oof_probs)
    
    # Save best threshold
    save_json({"threshold": best_thresh, "cv_f1": best_f1}, OUTPUTS_DIR / "best_threshold.json")
    
    # 4. Final Retraining
    logger.info("Retraining on full dataset...")
    
    # Feature Selection
    final_selector = FeatureSelector(output_report_path=OUTPUTS_DIR / "feature_selection_report.json")
    final_selector.fit(X, y)
    X_sel = final_selector.transform(X)
    
    # Preprocessing
    final_pipeline = PreprocessingPipeline(target_col=TARGET_COLUMN)
    final_pipeline.fit(X_sel, y)
    X_proc = final_pipeline.transform(X_sel)
    
    # Model
    final_model = None
    if args.model == "baseline":
        final_model = LogisticRegressionModel(**params)
    elif args.model == "rf":
        final_model = RandomForestModel(**params)
    elif args.model == "gbm":
        final_model = GradientBoostingModel(backend=args.gbm_backend, **params)

    cat_indices = final_pipeline.get_categorical_indices()
    
    if args.model == "gbm" and args.gbm_backend == "lightgbm":
         final_model.fit(X_proc, y.values, categorical_feature=cat_indices)
    else:
         final_model.fit(X_proc, y.values)
         
    # Save Artifacts
    save_pickle(final_selector, MODELS_DIR / "feature_selector.pkl")
    save_pickle(final_pipeline, MODELS_DIR / "preprocessing_pipeline.pkl")
    final_model.save(MODELS_DIR / f"{args.model}_model.pkl")
    
    # Save CV plots
    plot_roc_curve(y, oof_probs, OUTPUTS_DIR / "cv_roc_curve.png")
    
    logger.info("Training pipeline complete.")


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
    
    # Save results
    opt_dir = OUTPUTS_DIR / "optimization"
    results = optimizer.optimize_all(opt_dir)
    
    for model, params in results.items():
        out_file = opt_dir / f"best_params_{model}.json"
        save_json(params, out_file)
        logger.info(f"Saved best params for {model} to {out_file}")
    
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
    # Construct paths
    model_path = MODELS_DIR / f"{args.model}_model.pkl"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"
    selector_path = MODELS_DIR / "feature_selector.pkl"
    threshold_path = OUTPUTS_DIR / "best_threshold.json"
    
    run_inference(
        data_path=args.data_path,
        model_path=model_path,
        pipeline_path=pipeline_path,
        selector_path=selector_path,
        threshold_path=threshold_path,
        output_path=args.output_path,
        test_indexes_path=args.test_indexes
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
    parser_analyze = subparsers.add_parser("analyze", help="Run EDA")
    parser_analyze.add_argument("--data-path", type=str, required=True, help="Path to data csv")
    parser_analyze.add_argument("--metadata-path", type=str, required=True, help="Path to metadata csv")

    # Train
    p_train = subparsers.add_parser("train", help="Train a model")
    p_train.add_argument("--data-path", type=str, required=True, help="Path to training data CSV")
    p_train.add_argument("--model", type=str, choices=["baseline", "rf", "gbm"], required=True, help="Model to train")
    p_train.add_argument("--gbm-backend", type=str, choices=["lightgbm", "xgboost"], default="lightgbm", help="Backend for GBM model")
    p_train.add_argument("--params-path", type=str, help="Path to JSON file with optimized hyperparameters")
    p_train.add_argument("--subsample", type=int, help="Number of rows to subsample for debugging")

    # Optimize
    p_optimize = subparsers.add_parser("optimize", help="Run Hyperparameter Optimization")
    p_optimize.add_argument("--data-path", type=str, required=True, help="Path to training data CSV")
    p_optimize.add_argument("--model", type=str, choices=["baseline", "rf", "gbm", "all"], required=True, help="Model to optimize")
    p_optimize.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    p_optimize.add_argument("--gbm-backend", type=str, choices=["lightgbm", "xgboost"], default="lightgbm", help="Backend for GBM model")

    # Predict
    p_predict = subparsers.add_parser("predict", help="Run Inference")
    p_predict.add_argument("--test-indexes", type=str, help="Path to test indexes CSV (Optional, defaults to all data)")
    p_predict.add_argument("--data-path", type=str, default=str(DATA_DIR / "data.csv"), help="Path to full data CSV")
    p_predict.add_argument("--model", type=str, required=True, help="Model name (e.g. lgbm)")
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
