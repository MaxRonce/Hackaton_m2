import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

from ..models.random_forest import RandomForestModel
from ..models.gradient_boosting import GradientBoostingModel
from ..models.baseline import LogisticRegressionModel
from ..config import SEED
from ..utils.logger import setup_logger

logger = setup_logger("optimization.optuna")

class HyperparameterOptimizer:
    def __init__(self, models_to_optimize: List[str], X: np.ndarray, y: np.ndarray, n_trials: int = 50, gbm_backend: str = "lightgbm"):
        self.models_to_optimize = models_to_optimize
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.gbm_backend = gbm_backend
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
        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": SEED
            }
            model = RandomForestModel(**params)
        elif model_name == "gbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "random_state": SEED,
                "backend": self.gbm_backend
            }
            if self.gbm_backend == "lightgbm":
                 params["num_leaves"] = trial.suggest_int("num_leaves", 20, 100)
                 params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            elif self.gbm_backend == "xgboost":
                 params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                 # Add specific xgboost params if needed
            
            model = GradientBoostingModel(**params)
        elif model_name == "baseline":
            params = {
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2"]), # lbfgs supports l2
                "random_state": SEED
            }
            model = LogisticRegressionModel(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in cv.split(self.X, self.y):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            preds_proba = model.predict_proba(X_val)[:, 1]
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
