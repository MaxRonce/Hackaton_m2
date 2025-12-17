import pandas as pd
import numpy as np
from pathlib import Path
from ..utils.io import load_pickle
from ..data.loader import load_data
from ..config import ID_COLUMN

def run_inference(test_path: str, model_path: str, pipeline_path: str, output_path: str):
    """
    Run inference on test data.
    
    Args:
        test_path (str): Path to test CSV.
        model_path (str): Path to saved model pickle.
        pipeline_path (str): Path to saved preprocessing pipeline pickle.
        output_path (str): Path to save submission CSV.
    """
    # Load assets
    model = load_pickle(model_path)
    pipeline = load_pickle(pipeline_path)
    
    # Load data
    df_test = load_data(test_path)
    
    # Preprocess
    # Ensure ID column is preserved for output
    if ID_COLUMN not in df_test.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' not found in test data.")
        
    ids = df_test[ID_COLUMN]
    X_test = pipeline.transform(df_test)
    
    # Predict
    # The requirement is "id, prediction" where prediction is a probability?
    # "Output CSV strictly formatted as: id,prediction 123,0.8421"
    # This implies prediction probability of the positive class.
    
    preds_proba = model.predict_proba(X_test)[:, 1]
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": ids,
        "prediction": np.round(preds_proba, 4)
    })
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
