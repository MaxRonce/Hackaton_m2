import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from pathlib import Path

from ml_pipeline.models.base import BaseModel
from ml_pipeline.models.gradient_boosting import GradientBoostingModel

class GatedMoEModel(BaseModel):
    def __init__(self, component_indices: Dict[str, List[int]], gbm_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.component_indices = component_indices
        self.gbm_params = gbm_params if gbm_params else {"n_estimators": 100}
        
        # Identify specialists
        # Group 1: Subjective (Questionnaire, Demographics)
        self.subjective_comp = ["Questionnaire", "Demographics"]
        self.subjective_indices = []
        for c in self.subjective_comp:
            self.subjective_indices.extend(self.component_indices.get(c, []))
            
        # Group 2: Objective (Laboratory, Examination, etc.)
        self.objective_indices = []
        for c, indices in self.component_indices.items():
            if c not in self.subjective_comp and c != "Unknown":
                self.objective_indices.extend(indices)
                
        # Specialists
        self.expert_subjective = GradientBoostingModel(**self.gbm_params)
        self.expert_objective = GradientBoostingModel(**self.gbm_params)
        self.expert_global = GradientBoostingModel(**self.gbm_params)
        
        # Gating Network (Logistic Regression on Binary Mask of original components)
        self.gater = LogisticRegression(random_state=42)

    def _get_sparsity_mask(self, X: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask representing component availability for each sample.
        For each component, we check if ALL its features are missing (or if it's mostly missing).
        """
        n_samples = X.shape[0]
        # We'll use 2 features for the gater: subjective_mask and objective_mask
        mask = np.zeros((n_samples, 2))
        
        if self.subjective_indices:
            # Mask is 1 if any subjective data is present (transformed data usually imputes, 
            # so we'd need to check original NaN if possible, but here we scan for non-zero missing indicators if they exist)
            # Alternatively, if we don't have RAW info, we check if the expert's inputs are "reliable".
            # Simpler: The gater learns from the transformed data's variance or just works on full X.
            pass
            
        return mask

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit विशेषज्ञों and the router."""
        # 1. Train Professionals
        if self.subjective_indices:
            self.expert_subjective.fit(X[:, self.subjective_indices], y)
            
        if self.objective_indices:
            self.expert_objective.fit(X[:, self.objective_indices], y)
            
        self.expert_global.fit(X, y)
        
        # 2. Train Gater
        # The Gater takes the full X and learns to predict correct class by blending?
        # No, a simple Gating approach is often Stacked Generalization.
        # Let's use the probabilities of experts as features for the Gater (Stacking)
        prob_s = self.expert_subjective.predict_proba(X[:, self.subjective_indices])[:, 1] if self.subjective_indices else np.zeros(len(y))
        prob_o = self.expert_objective.predict_proba(X[:, self.objective_indices])[:, 1] if self.objective_indices else np.zeros(len(y))
        prob_g = self.expert_global.predict_proba(X)[:, 1]
        
        X_gate = np.column_stack([prob_s, prob_o, prob_g])
        self.gater.fit(X_gate, y)
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob_s = self.expert_subjective.predict_proba(X[:, self.subjective_indices])[:, 1] if self.subjective_indices else np.zeros(len(X))
        prob_o = self.expert_objective.predict_proba(X[:, self.objective_indices])[:, 1] if self.objective_indices else np.zeros(len(X))
        prob_g = self.expert_global.predict_proba(X)[:, 1]
        
        X_gate = np.column_stack([prob_s, prob_o, prob_g])
        return self.gater.predict_proba(X_gate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        return joblib.load(path)
