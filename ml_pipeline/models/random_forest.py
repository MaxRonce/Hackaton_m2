from sklearn.ensemble import RandomForestClassifier
import numpy as np
from .base import BaseModel
from ..config import SEED

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Reasonable defaults
        self.params.setdefault("n_estimators", 100)
        self.params.setdefault("max_depth", 10)
        self.params.setdefault("random_state", SEED)
        self.params.setdefault("n_jobs", -1)
        
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
