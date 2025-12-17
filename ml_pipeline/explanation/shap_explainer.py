import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ..models.base import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger("explanation.shap")

def run_shap_analysis(model: BaseModel, X_train: np.ndarray, X_test: np.ndarray, output_dir: str | Path, feature_names: list = None):
    """
    Run SHAP analysis and generate plots.
    
    Args:
        model (BaseModel): Trained model wrapper.
        X_train (np.ndarray): Training data (for background distribution).
        X_test (np.ndarray): Test data (to explain).
        output_dir (str | Path): Directory to save plots.
        feature_names (list): List of feature names.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Access underlying sklearn/lgb/xgb model
        inner_model = model.model
        
        # Determine explainer type
        explainer = None
        # Using a sample of background data to speed up kernel/exact explainers if needed, 
        # but TreeExplainer handles full data well usually. 
        # For KernelExplainer (if used), we'd need a summary (kmeans).
        
        # Check for Tree-based models (LGBM, XGB, RF)
        if hasattr(inner_model, "feature_importances_"): 
            # Most likely tree-based
            logger.info("Using TreeExplainer")
            explainer = shap.TreeExplainer(inner_model)
        elif hasattr(inner_model, "coef_"):
             # Linear model
             logger.info("Using LinearExplainer")
             explainer = shap.LinearExplainer(inner_model, X_train)
        else:
             logger.info("Using KernelExplainer (generic)")
             # Use a background summary for kernel explainer to be faster
             background = shap.kmeans(X_train, 10) if len(X_train) > 100 else X_train
             explainer = shap.KernelExplainer(inner_model.predict_proba, background)

        logger.info("Computing SHAP values...")
        # For some models (like Random Forest), shap_values is a list (one for each class).
        # We usually care about positive class (index 1).
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            # Binary classification usually returns list of 2 arrays
            shap_values = shap_values[1]
            
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png")
        plt.close()
        
        # Bar Plot (Global Importance)
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance_bar.png")
        plt.close()

        logger.info(f"SHAP plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to run SHAP analysis: {e}")
        # Don't raise, just log error so pipeline can continue? 
        # Or raise? Better to not crash the whole optimization if explanation fails.
        # But this is a dedicated explain command usually.
        raise
