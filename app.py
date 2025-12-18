import streamlit as st
import pandas as pd
import numpy as np
import sys
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml_pipeline.config import MODELS_DIR, DATA_DIR, TARGET_COLUMN, ID_COLUMN
from ml_pipeline.utils.io import load_pickle
from ml_pipeline.data.loader import load_data

# Page Config
st.set_page_config(page_title="Medical Profiler - Mortality Prediction", layout="wide", page_icon="🏥")

# Styles
st.markdown("""
<style>
    .big-font { font-size:24px !important; }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Medical Mortality Risk Profiler")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model_wrapper = load_pickle(MODELS_DIR / "gbm_model.pkl")
    pipeline = load_pickle(MODELS_DIR / "preprocessing_pipeline.pkl")
    selector = load_pickle(MODELS_DIR / "feature_selector.pkl")
    return model_wrapper, pipeline, selector

@st.cache_data
def load_dataset():
    # Load Main Data
    df = load_data(DATA_DIR / "data.csv")
    if ID_COLUMN in df.columns:
        df = df.set_index(ID_COLUMN)
    
    # Load Metadata
    try:
        meta = pd.read_csv(DATA_DIR / "features_metadata.csv")
        # 'index' is the code (e.g. AGQ130), 'SAS' is the descriptive label
        meta_dict = meta.set_index('index')['SAS'].to_dict()
    except Exception as e:
        st.warning(f"Metadata not found or invalid: {e}")
        meta_dict = {}
        
    return df, meta_dict

try:
    model_wrapper, pipeline, selector = load_assets()
    df, meta_dict = load_dataset()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()


# --- Sidebar: Patient Selection ---
st.sidebar.header("👤 Patient Selection")
patient_id = st.sidebar.selectbox("Select Patient Event (SEQN)", df.index.unique())

if st.sidebar.button("🎲 Random Patient"):
    patient_id = np.random.choice(df.index)
    st.experimental_rerun()

# Get Patient Data
patient_row = df.loc[[patient_id]].copy() # Keep as DF
true_label = patient_row[TARGET_COLUMN].values[0] if TARGET_COLUMN in patient_row.columns else None
if TARGET_COLUMN in patient_row.columns:
    patient_row = patient_row.drop(columns=[TARGET_COLUMN])

# --- Main Logic ---

# 1. Transform Data (Original)
try:
    X_sel = selector.transform(patient_row)
    X_proc = pipeline.transform(X_sel)
    
    # Predict
    prob = model_wrapper.predict_proba(X_proc)[0, 1]
except Exception as e:
    st.error(f"Processing failed: {e}")
    st.stop()

# --- Layout ---
col_pred, col_explain = st.columns([1, 2])

with col_pred:
    st.subheader("Risk Assessment")
    
    # Gauge / Metric
    delta_color = "inverse" if prob > 0.5 else "normal"
    st.metric("Mortality Risk Probability", f"{prob:.2%}", delta=None)
    
    if prob > 0.36: # Using optimized threshold
        st.markdown(f"<p class='big-font risk-high'>⚠️ HIGH RISK</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='big-font risk-low'>✅ LOW RISK</p>", unsafe_allow_html=True)
        
    if true_label is not None:
        st.info(f"Actual Outcome: {'† Deceased' if true_label==1 else 'Survived'}")

    st.write("---")
    st.write("### 🎛️ Simulation / What-If")
    st.caption("Modify top risk factors to see impact.")
    
    # Identify Top Features for this patient using SHAP
    # We need the Explainer. TreeExplainer is fast.
    explainer = shap.TreeExplainer(model_wrapper.model)
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
         shap_values = shap_values[1]
    
    # Get feature names
    def get_feature_names(pipeline):
        try:
            return pipeline.preprocessor.get_feature_names_out()
        except Exception as e:
            # Fallback Reconstruction if get_feature_names_out fails
            try:
                num_cols = pipeline.num_features # Using attributes from preprocessing.py
                cat_cols = pipeline.cat_features
                
                # Indicators
                indicators = []
                try:
                    imputer = pipeline.preprocessor.named_transformers_['num'].named_steps['imputer']
                    if hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
                        ind_features = imputer.indicator_.get_feature_names_out(num_cols)
                        indicators = list(ind_features)
                except:
                    pass
                    
                reconstructed = num_cols + indicators + cat_cols
                return reconstructed
            except:
                return [f"feat_{i}" for i in range(X_proc.shape[1])]

    feat_names = get_feature_names(pipeline)
    
    # Ensure dimensions match
    if len(feat_names) != X_proc.shape[1]:
        # Truncate or Pad
        if len(feat_names) > X_proc.shape[1]:
            feat_names = feat_names[:X_proc.shape[1]]
        else:
            feat_names += [f"feat_{i}" for i in range(len(feat_names), X_proc.shape[1])]

    # Create DF of features, values, shap, description
    # shap_values[0] is array of shape (n_features,)
    impact_df = pd.DataFrame({
        "feature": feat_names,
        "value": X_proc[0],
        # Check SHAP shape. 
        # If binary classification, shap_values might be (n_samples, n_features) or list.
        # We handled list above.
        "shap": shap_values[0] if shap_values.ndim > 1 else shap_values, # handle single sample array
        "abs_shap": np.abs(shap_values[0] if shap_values.ndim > 1 else shap_values)
    }).sort_values(by="abs_shap", ascending=False).head(10) # Top 10 drivers
    
    # Simulation Inputs
    simulated_row = patient_row.copy()
    
    st.markdown("##### 🔧 Adjust Key Risk Factors")
    
    for idx, row in impact_df.iterrows():
        feat_name = row['feature']
        
        # Clean feature name
        # Remove sklearn prefixes if present (e.g., 'num__', 'remainder__')
        raw_feat = feat_name.split("__")[-1] 
        # Check for our manual reconstruction prefixes
        if raw_feat.startswith("missing_"):
             continue # Skip editing missing indicators directly
             
        # Metadata description
        desc = meta_dict.get(raw_feat, "")
        
        # Check if this feature exists in original raw DF
        if raw_feat in patient_row.columns:
            # Input Type
            curr_val = patient_row[raw_feat].values[0]
            
            # Numeric or Cat?
            if pd.api.types.is_numeric_dtype(patient_row[raw_feat]):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                     st.write(f"**{raw_feat}**")
                     if desc: st.caption(desc)
                with col_b:
                    new_val = st.number_input(
                        f"Value for {raw_feat}", 
                        value=float(curr_val) if not pd.isna(curr_val) else 0.0,
                        label_visibility="collapsed",
                        key=f"sim_{raw_feat}"
                    )
                simulated_row[raw_feat] = new_val
            else:
                 # Try to show selectbox if few unique values?
                 st.text(f"{raw_feat}: {curr_val}")

    
    # Re-predict Button
    if st.button("🔄 Recalculate Risk"):
        sim_X_sel = selector.transform(simulated_row)
        sim_X_proc = pipeline.transform(sim_X_sel)
        sim_prob = model_wrapper.predict_proba(sim_X_proc)[0, 1]
        
        diff = sim_prob - prob
        st.metric("New Probability", f"{sim_prob:.2%}", delta=f"{diff:+.2%}", delta_color="inverse")


with col_explain:
    st.subheader("🔍 Explainability (SHAP)")
    
    # Waterfall Plot
    shap.initjs()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Map feature names to descriptive labels for the plot
    display_feat_names = []
    for f in feat_names:
        # Clean prefix (num__, cat__, etc.)
        clean_f = f.split("__")[-1]
        # Check if it's a missing indicator
        is_missing = "missingindicator_" in f.lower() or "missing_" in f.lower()
        
        # Get base variable name
        base_f = clean_f.replace("missingindicator_", "").replace("missing_", "")
        
        label = meta_dict.get(base_f, base_f)
        if is_missing:
            label = f"Missing: {label}"
        display_feat_names.append(label)

    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[0], 
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
        data=X_proc[0], 
        feature_names=display_feat_names
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    st.pyplot(fig)
    
    st.info("The Waterfall plot shows how each feature contributes to pushing the risk higher (Red) or lower (Blue) from the average.")

# --- Metadata Explorer ---
st.write("---")
with st.expander("📖 Data Dictionary"):
    st.dataframe(pd.DataFrame.from_dict(meta_dict, orient='index', columns=['Description']))
