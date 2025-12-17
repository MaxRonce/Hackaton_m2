import lightgbm as lgb
import xgboost as xgb
import numpy as np
from .base import BaseModel
from ..config import SEED

class GradientBoostingModel(BaseModel):
    def __init__(self, backend="lightgbm", **kwargs):
        """
        Args:
            backend (str): "lightgbm" or "xgboost"
            **kwargs: Model hyperparameters
        """
        super().__init__(**kwargs)
        self.backend = backend.lower()
        
        # Defaults
        self.params.setdefault("random_state", SEED)
        
        if self.backend == "lightgbm":
            self.params.setdefault("objective", "binary")
            self.params.setdefault("metric", "auc")
            self.params.setdefault("verbosity", -1)
            self.model = lgb.LGBMClassifier(**self.params)
            
        elif self.backend == "xgboost":
            self.params.setdefault("objective", "binary:logistic")
            self.params.setdefault("eval_metric", "auc")
            self.params.setdefault("verbosity", 0)
            self.model = xgb.XGBClassifier(**self.params)
            
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingModel':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
