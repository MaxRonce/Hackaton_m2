from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
from typing import Dict

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        
    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
    return metrics
