
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.utils.io import load_pickle
from ml_pipeline.config import MODELS_DIR
import pandas as pd
import numpy as np

def debug_pipeline():
    try:
        pipeline = load_pickle(MODELS_DIR / "preprocessing_pipeline.pkl")
        print("\n--- PreprocessingPipeline ---")
        print(f"Type: {type(pipeline)}")
        print(f"Has preprocessor? {hasattr(pipeline, 'preprocessor')}")
        
        if hasattr(pipeline, "preprocessor"):
            preproc = pipeline.preprocessor
            print(f"\n--- ColumnTransformer ---")
            print(f"Transformers: {[t[0] for t in preproc.transformers]}")
            
            try:
                names = preproc.get_feature_names_out()
                print(f"\nFeature Names Out (First 10): {names[:10]}")
                print(f"Total Names: {len(names)}")
            except Exception as e:
                print(f"\nError getting feature names: {e}")

        print(f"\n--- Pipeline Attributes ---")
        if hasattr(pipeline, "numerical_features_"):
             print(f"Num Features Count: {len(pipeline.numerical_features_)}")
             # print(f"Num Features (First 5): {pipeline.numerical_features_[:5]}")
        else:
             print("Pipeline has no 'numerical_features_' attribute")

        if hasattr(pipeline, "categorical_features_"):
             print(f"Cat Features Count: {len(pipeline.categorical_features_)}")
        else:
             print("Pipeline has no 'categorical_features_' attribute")

        selector = load_pickle(MODELS_DIR / "feature_selector.pkl")
        print("\n--- FeatureSelector ---")
        if hasattr(selector, "selected_features_"):
            print(f"Selected Features Count: {len(selector.selected_features_)}")
            
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    debug_pipeline()
