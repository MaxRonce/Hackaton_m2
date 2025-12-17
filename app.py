import streamlit as st
import pandas as pd
import numpy as np
import sys
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml_pipeline.config import MODELS_DIR, DATA_DIR, TARGET_COLUMN
from ml_pipeline.utils.io import load_pickle
from ml_pipeline.data.loader import load_data
from ml_pipeline.models.base import BaseModel
from ml_pipeline.models.ensemble import SoftVotingEnsemble, load_models_from_paths

st.set_page_config(page_title="Hackaton Machine Learning Pipeline", layout="wide")

st.title("Hackaton Machine Learning Pipeline")

tabs = st.tabs(["📊 EDA", "🧠 Inference", "⚙️ Model Info"])

@st.cache_data
def load_data_cached(path):
    return load_data(path)

with tabs[0]:
    st.header("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload CSV for Analysis", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        st.write("### Statistics")
        st.write(df.describe())
        
        st.write("### Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select Numeric Column", numeric_cols)
            
            if selected_col:
                fig, ax = plt.subplots()
                sns.histplot(df[selected_col], kde=True, ax=ax)
                st.pyplot(fig)
            
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            selected_cat = st.selectbox("Select Categorical Column", categorical_cols)
            
            if selected_cat:
                fig, ax = plt.subplots()
                sns.countplot(x=selected_cat, data=df, ax=ax)
                st.pyplot(fig)

with tabs[1]:
    st.header("Real-time Inference")
    
    # Load available models
    model_files = list(MODELS_DIR.glob("*_model.pkl"))
    model_names = [f.name for f in model_files]
    model_names.append("Ensemble (All Models)")
    
    if not model_files:
        st.warning("No trained models found in outputs/models/. Please train a model first (e.g., `python ml_pipeline/main.py train ...`).")
    else:
        selected_model_name = st.selectbox("Select Model", model_names)
        
        # Inputs
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Pclass", [1, 2, 3])
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.number_input("Age", 0, 100, 30)
            sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
        
        with col2:
            parch = st.number_input("Parents/Children", 0, 10, 0)
            fare = st.number_input("Fare", 0.0, 500.0, 32.0)
            embarked = st.selectbox("Embarked", ["S", "C", "Q"])
            
        # Create DataFrame
        input_data = pd.DataFrame({
            "passengerid": [9999], # Dummy
            "pclass": [pclass],
            "name": ["Unknown"],
            "sex": [sex],
            "age": [age],
            "sibsp": [sibsp],
            "parch": [parch],
            "ticket": ["XXXX"],
            "fare": [fare],
            "cabin": [np.nan],
            "embarked": [embarked]
        })
        
        if st.button("Predict Survival Probability"):
            try:
                pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"
                pipeline = load_pickle(pipeline_path)
                
                # Preprocess
                # Note: Pipeline expects standard columns. We constructed them above.
                X_transformed = pipeline.transform(input_data)
                
                # Load Model(s)
                model = None
                if selected_model_name == "Ensemble (All Models)":
                     # Load all available models
                     paths = [str(p) for p in MODELS_DIR.glob("*_model.pkl")]
                     model = load_models_from_paths(paths)
                     st.info(f"Ensembling {len(paths)} models: {[Path(p).name for p in paths]}")
                else:
                    model_path = MODELS_DIR / selected_model_name
                    model = load_pickle(model_path)
                
                # Predict
                prob = model.predict_proba(X_transformed)
                if prob.ndim == 2:
                    prob = prob[0, 1]
                elif prob.ndim == 1:
                    prob = prob[0]
                
                st.success(f"Survival Probability: **{prob:.4f}**")
                
                if prob > 0.5:
                    st.write("🎉 Likely to Survive")
                else:
                    st.write("💀 Unlikely to Survive")

                # SHAP Explanation (Only for single models roughly, Ensemble is hard)
                if selected_model_name != "Ensemble (All Models)":
                    st.subheader("Why this prediction?")
                    with st.spinner("Calculating SHAP values..."):
                        # We need the inner model
                        inner_model = model.model
                        
                        explainer = None
                        if hasattr(inner_model, "feature_importances_"): 
                             explainer = shap.TreeExplainer(inner_model)
                        elif hasattr(inner_model, "coef_"):
                             # Linear models might need background data for some plots or be simpler
                             # For simplicity let's skip linear SHAP in app or handle differently
                             st.warning("SHAP explanation supported best for Tree models (RF, GBM) in this app.")
                             explainer = None
                        
                        if explainer:
                            shap_values = explainer.shap_values(X_transformed)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]
                                
                            # Force Plot
                            st.write("Force Plot")
                            # We use matplotlib to render
                            shap.initjs()
                            
                            # Waterfall is often better for single instance
                            try:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                # shap.waterfall_plot expects Explanation object usually for new shap versions
                                # or shap_values[0] if array.
                                # Let's create an Explanation object to be safe with modern shap
                                explanation = shap.Explanation(shap_values[0], base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, data=X_transformed[0], feature_names=pipeline.pipeline.get_feature_names_out())
                                shap.plots.waterfall(explanation, show=False)
                                st.pyplot(plt.gcf())
                            except Exception as e:
                                st.warning(f"Could not render Waterfall plot: {e}")
                                # Fallback
                                st.write("Feature Values for this instance:")
                                st.write(pd.DataFrame(X_transformed, columns=pipeline.pipeline.get_feature_names_out()).T)

                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.text(traceback.format_exc())

with tabs[2]:
    st.header("Model Performance")
    # Load metrics jsons
    metric_files = list(Path("outputs").glob("*_metrics.json"))
    if metric_files:
        for f in metric_files:
            import json
            with open(f) as json_file:
                metrics = json.load(json_file)
            st.write(f"**{f.name}**")
            st.json(metrics)
            
    # Show comparison plot if exists
    comparison_plot = Path("outputs/models/optimization_comparison.png")
    if comparison_plot.exists():
        st.image(str(comparison_plot), caption="Optimization Comparison")
