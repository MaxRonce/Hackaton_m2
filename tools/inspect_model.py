
import sys
from pathlib import Path
# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_pipeline.utils.io import load_pickle

def inspect_model():
    model_path = Path("outputs/models/gbm_model.pkl")
    pipeline_path = Path("outputs/models/preprocessing_pipeline.pkl")
    
    if not model_path.exists():
        print("Model not found.")
        return

    print(f"Loading model from {model_path}...")
    model_wrapper = load_pickle(model_path)
    model = model_wrapper.model  # Access underlying lightgbm Booster/sklearn wrapper
    
    print(f"Loading pipeline from {pipeline_path}...")
    pipeline = load_pickle(pipeline_path)
    
    # Get Feature Names from Pipeline
    # Pipeline is ColumnTransformer -> 'numeric', 'categorical'
    # logical names are needed.
    # PreprocessingPipeline has a 'get_feature_names_out' method?
    # Or strict mapping.
    # Let's try standard sklearn method
    try:
        feature_names = pipeline.get_feature_names_out()
    except AttributeError:
        print("Could not get feature names from pipeline directly.")
        feature_names = [f"feat_{i}" for i in range(model.n_features_in_)]

    print(f"Number of Features: {len(feature_names)}")
    

    # ... previous imports ...
    from ml_pipeline.explanation.shap_explainer import run_shap_analysis
    from ml_pipeline.data.loader import load_data
    from ml_pipeline.config import TARGET_COLUMN, ID_COLUMN

    # ... load model and pipeline ...
    
    # Load Data for SHAP (Need transformed data)
    print("Loading data for SHAP analysis...")
    df = load_data("data/data.csv")
    if ID_COLUMN in df.columns:
        df = df.set_index(ID_COLUMN)
    
    # Feature Selection (Need selector)
    selector_path = Path("outputs/models/feature_selector.pkl")
    print(f"Loading selector from {selector_path}...")
    selector = load_pickle(selector_path)
    
    # Transform
    X = df.drop(columns=[TARGET_COLUMN], errors='ignore')
    # If merged previously, target might be there. If prediction file, maybe not.
    # Training data definitely has it.
    
    X_sel = selector.transform(X)
    X_proc = pipeline.transform(X_sel)
    
    # For SHAP, we need a sample if data is huge, but 60k is okay-ish for TreeExplainer.
    # Let's take a sample of 2000 for speed if needed, or full if user wants detail.
    # User said "vrai expliquabilité", let's use a robust sample.
    sample_size = 2000
    if len(X_proc) > sample_size:
        print(f"Sampling {sample_size} rows for SHAP...")
        # Use random choice
        indices = np.random.choice(X_proc.shape[0], sample_size, replace=False)
        X_shap = X_proc[indices]
    else:
        X_shap = X_proc
        
    # Feature Names
    # Try to get from pipeline
    try:
        # PreprocessingPipeline might not expose get_feature_names_out directly if custom
        # But step 'preprocessor' (ColumnTransformer) does.
        feature_names = pipeline.preprocessor.get_feature_names_out()
    except:
        print("Could not get feature names from pipeline, using generic.")
        feature_names = [f"feature_{i}" for i in range(X_shap.shape[1])]
        
    # Run SHAP
    print("Running SHAP Analysis...")
    run_shap_analysis(
        model=model_wrapper, 
        X_train=X_shap, # TreeExplainer uses this for expected value logic sometimes, or just X
        X_test=X_shap, 
        output_dir="outputs/eda",
        feature_names=feature_names
    )
    
    # Still dump gain-based importance as backup
    # ...


if __name__ == "__main__":
    inspect_model()
