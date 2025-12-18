import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from typing import List, Tuple, Optional
from ..utils.logger import setup_logger
from ..config import ID_COLUMN, TARGET_COLUMN, DROP_COLUMNS

logger = setup_logger("data.preprocessing")

class PreprocessingPipeline:
    def __init__(self, target_col: str = TARGET_COLUMN, id_col: str = ID_COLUMN, drop_cols: Optional[List[str]] = DROP_COLUMNS):
        self.target_col = target_col
        self.id_col = id_col
        self.drop_cols = drop_cols if drop_cols else []
        self.pipeline = None
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """
        Fit the preprocessing pipeline.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Target vector.
        """
        X_clean = self._drop_features(X)
        self.feature_names_in_ = X_clean.columns.tolist() # Keep track of columns
        
        # Identify column types
        self.num_features = X_clean.select_dtypes(include=['number']).columns.tolist()
        self.cat_features = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical features: {self.num_features}")
        logger.info(f"Categorical features: {self.cat_features}")
        
        # Define transformers
        # Numeric: Median imputation + Missing Indicator + Robust Scaling (better for outliers)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()) # RobustScaler might be better but let's stick to Standard or None for now as per prior plan mention? 
                                         # User Prompt said "Scaling standard" was risky, proposed "Median imputation (after filtering)".
                                         # Let's use RobustScaler as I planned in "implementation_plan.md" (Optional clipping... I said RobustScaler in plan text)
                                         # Actually, I'll use RobustScaler.
        ])
        from sklearn.preprocessing import RobustScaler
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', RobustScaler()) 
        ])
        
        # Categorical: explicit MISSING, Ordinal Encoding for LightGBM
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ],
            verbose_feature_names_out=False # Clean names
        )
        
        self.pipeline.fit(X_clean, y)
        logger.info("Preprocessing pipeline fitted.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data.
        
        Args:
            X (pd.DataFrame): Input features.
            
        Returns:
            np.ndarray: Processed numpy array.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        X_clean = self._drop_features(X)
        
        # Verify columns match training time (handling missing columns via alignment if needed, 
        # but sklearn column transformer expects exact match or subset if configured strictly,
        # usually user ensures schema consistency).
        missing_cols = set(self.feature_names_in_) - set(X_clean.columns)
        if missing_cols:
             logger.warning(f"Missing columns in input: {missing_cols}. This might cause errors.")

        X_transformed = self.pipeline.transform(X_clean)
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to drop ID and ignored columns."""
        cols_to_drop = [c for c in self.drop_cols + [self.id_col] if c in df.columns]
        if self.target_col in df.columns:
            cols_to_drop.append(self.target_col)
            
        return df.drop(columns=cols_to_drop)


    def get_categorical_indices(self) -> List[int]:
        """
        Returns indices of categorical columns in the transformed array.
        Assumes 'num' then 'cat' order in ColumnTransformer.
        """
        if self.pipeline is None:
            return []
        
        # In ColumnTransformer, output is concatenated in order of transformers
        n_num = len(self.num_features)
        n_cat = len(self.cat_features)
        
        # Cat indices start after Num indices
        return list(range(n_num, n_num + n_cat))
