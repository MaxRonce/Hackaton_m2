from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger("models.base")

class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass

    def save(self, path: str | Path):
        """Save the model to disk."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    @classmethod
    def load(cls, path: str | Path) -> 'BaseModel':
        """Load the model from disk."""
        try:
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
