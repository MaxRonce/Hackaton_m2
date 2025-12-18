
import numpy as np
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)

def optimize_threshold(y_true, y_prob):
    """
    Finds the best threshold for F1 score.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    logger.info(f"Threshold Optimization: Best F1={best_f1:.4f} at threshold={best_thresh:.2f}")
    return best_thresh, best_f1
