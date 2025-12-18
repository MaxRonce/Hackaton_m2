import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

from ..models.random_forest import RandomForestModel
from ..models.gradient_boosting import GradientBoostingModel
from ..models.baseline import LogisticRegressionModel
from ..feature_filtering import FeatureSelector
from ..data.preprocessing import PreprocessingPipeline
from ..config import SEED, TARGET_COLUMN
from ..utils.logger import setup_logger

logger = setup_logger("optimization.optuna")

class HyperparameterOptimizer:
    def __init__(self, models_to_optimize: List[str], X: pd.DataFrame, y: pd.Series, n_trials: int = 50, gbm_backend: str = "lightgbm", metadata_path: Optional[str] = None):
        self.models_to_optimize = models_to_optimize
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.gbm_backend = gbm_backend
        self.metadata_path = metadata_path
        self.best_scores = {}

    def optimize_all(self, output_dir: str | Path) -> Dict[str, Any]:
        """Run optimization for all specified models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for model_name in self.models_to_optimize:
            logger.info(f"Optimizing {model_name}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._objective(trial, model_name), n_trials=self.n_trials)
            
            results[model_name] = study.best_trial.params
            self.best_scores[model_name] = study.best_value
            logger.info(f"Best params for {model_name}: {study.best_trial.params} (Score: {study.best_value:.4f})")
            
        self._plot_results(output_dir)
        return results

    def _objective(self, trial, model_name):
        # 1. Feature Selection Hyperparameters
        fs_params = {
            "max_missing_rate": trial.suggest_float("max_missing_rate", 0.5, 0.95),
            "variance_threshold": trial.suggest_float("variance_threshold", 0.0, 0.5)
        }
        
        # 2. Model Hyperparameters (Suggest only)
        model_params = {}
        if model_name == "rf":
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": SEED
            }
        elif model_name == "gbm" or model_name == "moe":
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "random_state": SEED,
                "backend": self.gbm_backend
            }
            if self.gbm_backend == "lightgbm":
                 model_params["num_leaves"] = trial.suggest_int("num_leaves", 20, 100)
                 model_params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            elif self.gbm_backend == "xgboost":
                 model_params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        elif model_name == "baseline":
            model_params = {
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "penalty": "l2",
                "random_state": SEED
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 3. Cross-Validation with Feature Selection and Preprocessing inside the loop
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        
        for train_idx, val_idx in cv.split(self.X, self.y):
            X_train_raw, X_val_raw = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # A. Feature Filtering
            selector = FeatureSelector(
                max_missing_rate=fs_params["max_missing_rate"],
                variance_threshold=fs_params["variance_threshold"],
                output_report_path=None
            )
            selector.fit(X_train_raw, y_train)
            X_train_sel = selector.transform(X_train_raw)
            X_val_sel = selector.transform(X_val_raw)
            
            # B. Preprocessing
            pipeline = PreprocessingPipeline(target_col=TARGET_COLUMN)
            pipeline.fit(X_train_sel, y_train)
            X_train_proc = pipeline.transform(X_train_sel)
            X_val_proc = pipeline.transform(X_val_sel)
            
            # C. Model Instantiation (Inside loop to handle varying feature counts)
            from ..models.moe import GatedMoEModel
            if model_name == "rf":
                model = RandomForestModel(**model_params)
            elif model_name == "gbm":
                model = GradientBoostingModel(**model_params)
            elif model_name == "moe":
                metadata = pd.read_csv(self.metadata_path) if self.metadata_path else None
                comp_indices = pipeline.get_component_feature_indices(metadata) if metadata is not None else {}
                model = GatedMoEModel(component_indices=comp_indices, gbm_params=model_params)
            elif model_name == "baseline":
                model = LogisticRegressionModel(**model_params)
            
            # D. Fit & Predict
            # For LightGBM, we might want categorical indices
            if model_name == "gbm" and self.gbm_backend == "lightgbm":
                cat_indices = pipeline.get_categorical_indices()
                model.fit(X_train_proc, y_train.values, categorical_feature=cat_indices)
            else:
                model.fit(X_train_proc, y_train.values)
                
            preds_proba = model.predict_proba(X_val_proc)[:, 1]
            try:
                scores.append(roc_auc_score(y_val, preds_proba))
            except:
                return 0.0

        return np.mean(scores)

    def _plot_results(self, output_dir: Path):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(self.best_scores.keys()), y=list(self.best_scores.values()))
        plt.title(f"Best ROC-AUC Scores (Optimization)")
        plt.ylim(0, 1)
        plt.savefig(output_dir / "optimization_comparison.png")
        plt.close()
