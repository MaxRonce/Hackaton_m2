from sklearn.linear_model import LogisticRegression
import numpy as np
from .base import BaseModel
from ..config import SEED

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set defaults if not provided
        self.params.setdefault("random_state", SEED)
        self.params.setdefault("max_iter", 1000)
        self.params.setdefault("solver", "lbfgs")
        
        self.model = LogisticRegression(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionModel':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
