import numpy as np
import pandas as pd
from typing import List
from ..utils.logger import setup_logger
from ..utils.io import load_pickle

logger = setup_logger("models.ensemble")

class SoftVotingEnsemble:
    def __init__(self, models: List[object]):
        """
        Args:
            models: List of loaded model objects (must have predict_proba)
        """
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Average probabilities from all models.
        """
        probas = []
        for i, model in enumerate(self.models):
            try:
                p = model.predict_proba(X)
                # Ensure we take the positive class probability if shape is (N, 2)
                if p.ndim == 2 and p.shape[1] == 2:
                    p = p[:, 1]
                probas.append(p)
            except Exception as e:
                logger.error(f"Model {i} failed to predict: {e}")
                raise
        
        # Stack and average
        probas = np.column_stack(probas)
        avg_proba = np.mean(probas, axis=1)
        return avg_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

def load_models_from_paths(paths: List[str]) -> 'SoftVotingEnsemble':
    loaded_models = []
    for p in paths:
        model = load_pickle(p)
        loaded_models.append(model)
    return SoftVotingEnsemble(loaded_models)
