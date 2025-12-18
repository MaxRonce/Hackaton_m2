
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import json
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on:
    1. Missing value percentage (default < 95%)
    2. Variance (removes near-zero variance)
    3. (Optional) Minimum samples per class if y is provided.
    
    Generates a report of dropped features.
    """
    
    def __init__(self, 
                 max_missing_rate: float = 0.95, 
                 variance_threshold: float = 0.0,
                 min_samples_per_class: int = 5,
                 output_report_path: Optional[Union[str, Path]] = None):
        self.max_missing_rate = max_missing_rate
        self.variance_threshold = variance_threshold
        self.min_samples_per_class = min_samples_per_class
        self.output_report_path = output_report_path
        
        self.dropped_features_ = {}
        self.selected_features_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Identifies features to drop.
        """
        logger.info("Fitting FeatureSelector...")
        n_rows, n_cols = X.shape
        self.dropped_features_ = {
            "missing_rate": [],
            "low_variance": [],
            "low_samples_per_class": []
        }
        
        # 1. Missing Rate
        missing_rate = X.isnull().mean()
        drop_missing = missing_rate[missing_rate > self.max_missing_rate].index.tolist()
        self.dropped_features_["missing_rate"] = drop_missing
        logger.info(f"Found {len(drop_missing)} features with > {self.max_missing_rate:.1%} missing values.")
        
        # Features to check for variance (exclude already droppped)
        remaining_cols = [c for c in X.columns if c not in drop_missing]
        
        # 2. Variance (Near Zero)
        # Note: Variance calculation ignores NaNs by default in pandas
        variances = X[remaining_cols].var()
        drop_variance = variances[variances <= self.variance_threshold].index.tolist()
        
        # Also check for single unique value (if not caught by var=0, e.g. due to all NaNs except 1 value?)
        # Actually pd.var() of single value is NaN. pd.var() of constant is 0.
        # Let's also check for columns with only 1 unique value observed (ignoring NaNs)
        # explicitly as "low information"
        for col in remaining_cols:
            if col in drop_variance:
                continue
            if X[col].nunique(dropna=True) <= 1:
                drop_variance.append(col)
                
        self.dropped_features_["low_variance"] = list(set(drop_variance))
        logger.info(f"Found {len(self.dropped_features_['low_variance'])} features with low variance.")
        
        remaining_cols = [c for c in remaining_cols if c not in self.dropped_features_["low_variance"]]
        
        # 3. Low samples per class (if y provided)
        if y is not None:
            # Only relevant for binary/multiclass classification
            # Drop features that have too few non-null observations for a specific class
            # This is expensive to check loop-wise for 1500 features? Not really for 1.5k cols.
            # Vectorized approach:
            
            # This is custom logic: "Supprimer les features avec trop peu d’observations non-nulles dans chaque classe"
            # Interpreted as: if for any class C, count(X[col] is not null & y==C) < thresh -> drop?
            # Or if sum of non-nulls in ALL classes is low? The prompt says "dans chaque classe".
            # Assume: If count of non-nulls in Class 0 < T OR count in Class A < T -> Drop.
            
            classes = y.unique()
            drop_class_sparsity = []
            
            for col in remaining_cols:
                mask_not_null = X[col].notna()
                valid = True
                for label in classes:
                    count = (mask_not_null & (y == label)).sum()
                    if count < self.min_samples_per_class:
                        valid = False
                        break
                if not valid:
                    drop_class_sparsity.append(col)
            
            self.dropped_features_["low_samples_per_class"] = drop_class_sparsity
            logger.info(f"Found {len(drop_class_sparsity)} features with low samples per class.")
            
        # Compile selected
        all_dropped = set(self.dropped_features_["missing_rate"] + 
                          self.dropped_features_["low_variance"] + 
                          self.dropped_features_["low_samples_per_class"])
        
        self.selected_features_ = [c for c in X.columns if c not in all_dropped]
        logger.info(f"Feature Selection complete. Dropped {len(all_dropped)} features. Kept {len(self.selected_features_)}.")
        
        if self.output_report_path:
            self._save_report(n_cols, len(all_dropped))
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the features.
        """
        if not self.selected_features_:
            # If fitted but everything dropped or not fitted well? 
            # Should have at least one feature.
            # If not fitted, return X (sklearn standard behavior usually requires check_is_fitted)
            logger.warning("FeatureSelector: No features selected or not fitted. Returning X.")
            return X
            
        # Return only selected features that exist in X
        # (Handle case where test set might miss some cols if they were dropped anyway? No, X must have them)
        # We assume X has the columns.
        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        return X[cols_to_keep]
    
    def _save_report(self, n_total, n_dropped):
        report = {
            "n_total_features": n_total,
            "n_dropped_features": n_dropped,
            "n_kept_features": len(self.selected_features_),
            "details": self.dropped_features_
        }
        
        path = Path(self.output_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Feature selection report saved to {path}")

