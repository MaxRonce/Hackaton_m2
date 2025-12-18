import pandas as pd
import numpy as np
from pathlib import Path
from ..utils.io import load_pickle
from ..data.loader import load_data
from ..config import ID_COLUMN, TARGET_COLUMN

from ..utils.io import load_json

from typing import Optional
import logging

logger = logging.getLogger(__name__)

def run_inference(data_path: str, model_path: str, pipeline_path: str, selector_path: str, threshold_path: str, output_path: str, test_indexes_path: Optional[str] = None):
    """
    Run inference on data.
    """
    # 1. Load Assets
    model = load_pickle(model_path)
    pipeline = load_pickle(pipeline_path)
    selector = load_pickle(selector_path)
    threshold_data = load_json(threshold_path)
    best_thresh = threshold_data["threshold"]
    
    # 2. Load Data
    df = load_data(data_path)
    if ID_COLUMN not in df.columns:
         raise ValueError(f"ID column {ID_COLUMN} missing in data.")
    
    df_indexed = df.set_index(ID_COLUMN)
    
    # Determine Target IDs
    if test_indexes_path:
        test_idx_df = pd.read_csv(test_indexes_path, header=None)
        target_ids = test_idx_df.iloc[:, 0].values
        logger.info(f"Predicting on {len(target_ids)} IDs from {test_indexes_path}")
        
        # Reindex to target_ids (Force alignment)
        X_test_aligned = df_indexed.reindex(target_ids)
        
        # Check for missing data
        missing_ids_count = X_test_aligned.isnull().all(axis=1).sum()
        if missing_ids_count > 0:
            logger.warning(f"WARNING: {missing_ids_count} requested IDs were not found in data.csv. They will be imputed as fully missing.")
            
    else:
        # Full prediction
        logger.info(f"Predicting on all {len(df)} rows in {data_path}")
        target_ids = df_indexed.index.values
        X_test_aligned = df_indexed
        
    # 4. Transform
    # Feature Selection
    X_sel = selector.transform(X_test_aligned)
    
    # Preprocessing
    X_proc = pipeline.transform(X_sel)
    
    # 5. Predict
    preds_proba = model.predict_proba(X_proc)[:, 1]
    
    # Apply Threshold
    preds_label = (preds_proba >= best_thresh).astype(int)
    
    # 6. Save
    # "Expected 59064 labels". User wants one column? Or ID and Prediction?
    # User comment: "Output saved to outputs/predictions.csv. File Format error Expected 59064 labels got 2954 instead !"
    # The previous code saved "prediction" column only (from `pd.DataFrame(preds_label, columns=[TARGET_COLUMN])`).
    # If the user wants to be safe, I should probably save ID and Prediction if the format allows, 
    # but the user initially said "exactly one column of predictions".
    # I'll stick to one column BUT predict on ALL rows.
    
    submission = pd.DataFrame(preds_label, columns=[TARGET_COLUMN])
    submission.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path} with threshold {best_thresh:.4f}")
        
    # 4. Transform
    # Feature Selection
    X_sel = selector.transform(X_test_aligned)
    
    # Preprocessing
    X_proc = pipeline.transform(X_sel)
    
    # 5. Predict
    preds_proba = model.predict_proba(X_proc)[:, 1]
    
    # Apply Threshold
    preds_label = (preds_proba >= best_thresh).astype(int)
    
    submission = pd.DataFrame(preds_label, columns=[TARGET_COLUMN]) # Using config target name 'y'
    
    # Write to CSV
    submission.to_csv(output_path, index=False)

    # Edit the 1st row of
    
    print(f"Predictions saved to {output_path} with threshold {best_thresh:.4f}")

