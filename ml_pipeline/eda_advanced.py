
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Optional, List, Dict
from ml_pipeline.config import ID_COLUMN, TARGET_COLUMN

logger = logging.getLogger(__name__)

# --- Premium Styling ---
def set_premium_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        'figure.autolayout': True,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'font.family': 'sans-serif'
    })
    sns.set_palette("viridis")

def run_advanced_eda(data_path: Path, metadata_path: Path, output_dir: Path):
    """
    Runs advanced EDA including sparsity analysis, metadata-driven analysis,
    and target-conditional distributions.
    """
    logger.info("Starting Advanced EDA...")
    
    # Load Data
    data = pd.read_csv(data_path)
    metadata = pd.read_csv(metadata_path)
    
    # Metadata mapping
    meta_dict = metadata.set_index('index')['SAS'].to_dict()
    
    set_premium_style()
    
    
    # Try to load ground_truth_train.csv from same dir as data_path
    ground_truth_path = data_path.parent / "ground_truth_train.csv"
    if ground_truth_path.exists():
        labels = pd.read_csv(ground_truth_path)
        if ID_COLUMN in data.columns and ID_COLUMN in labels.columns:
            # We want to keep all data for sparsity analysis, but add labels where available?
            # Or restricted to labeled set?
            # User said "Objectif: Justifier pourquoi tu supprimes... Analyse par type...".
            # Metadata analysis on full data is better. Target conditional needs target.
            # I will merge with 'left' join to keep all features data, and have NaN target for unlabeled?
            # But the ground_truth is 95% of data.
            # If I do inner join, I lose 5% (test).
            # If I do left, `y` will be NaN for test.
            # `X` should be all data. `y` only valid where present.
            data = data.merge(labels, on=ID_COLUMN, how="left")
            logger.info("Merged with ground truth labels for EDA.")
    
    # Ensure ID_COLUMN and TARGET_COLUMN are handled
    if ID_COLUMN in data.columns:
        data = data.set_index(ID_COLUMN)

        
    # Separate features and target if target exists
    if TARGET_COLUMN in data.columns:
        y = data[TARGET_COLUMN]
        X = data.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = data
        
    # Create output directories
    sparsity_dir = output_dir / "sparsity"
    groups_dir = output_dir / "by_group"
    target_dir = output_dir / "target_conditional"
    
    for d in [sparsity_dir, groups_dir, target_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    summary_report = {}

    # 1. Sparsity Analysis
    logger.info("Running Sparsity Analysis...")
    summary_report["sparsity"] = _analyze_sparsity(X, sparsity_dir, meta_dict)

    # 2. Metadata Analysis
    logger.info("Running Metadata Analysis...")
    summary_report["metadata_groups"] = _analyze_metadata(X, metadata, groups_dir, meta_dict)

    # 3. Target Conditional Analysis
    if y is not None:
        logger.info("Running Target Conditional Analysis...")
        _analyze_target_conditional(X, y, target_dir, meta_dict)
        
    # 4. Filtering Potential Analysis (Justification)
    logger.info("Running Filtering Potential Analysis...")
    summary_report["filtering_potential"] = _analyze_filtering_potential(X, output_dir, meta_dict)
        
    # Save Summary Report
    with open(output_dir / "summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=4)
        
    logger.info(f"Advanced EDA completed. Results saved to {output_dir}")


def _analyze_sparsity(X: pd.DataFrame, output_dir: Path, meta_dict: Dict) -> Dict:
    """
    Analyzes missing values: % per feature, % per row, histograms, heatmap.
    """
    n_rows, n_cols = X.shape
    
    # Feature Missingness
    missing_by_feature = X.isnull().mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(missing_by_feature * 100, bins=50, kde=True, color='teal')
    plt.axvline(80, color='red', linestyle='--', label='Selection Threshold (80%)')
    plt.title("Distribution of Missing Values % per Feature")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Count of Features")
    plt.legend()
    plt.savefig(output_dir / "missing_per_feature_hist.png")
    plt.close()
    
    # Top 20 Most Sparse Features (Justification)
    top_sparse = missing_by_feature.head(20)
    plt.figure(figsize=(14, 10))
    sparse_labels = [meta_dict.get(idx, idx)[:50] + "..." if len(meta_dict.get(idx, idx)) > 50 else meta_dict.get(idx, idx) for idx in top_sparse.index]
    sns.barplot(x=top_sparse.values * 100, y=sparse_labels, hue=sparse_labels, palette="magma", legend=False)
    plt.title("Top 20 Most Sparse Features (Candidates for Removal)")
    plt.xlabel("Missing Values (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "top_sparse_features.png")
    plt.close()
    
    # Row Missingness
    missing_by_row = X.isnull().mean(axis=1)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(missing_by_row * 100, bins=50, kde=True, color='orange')
    plt.title("Distribution of Missing Values % per Individual")
    plt.xlabel("Data Missing (%)")
    plt.ylabel("Count of Individuals")
    plt.savefig(output_dir / "missing_per_row_hist.png")
    plt.close()
    
    # Sparsity Heatmap (sampled if too large)
    sample_size = min(1000, n_rows)
    X_sample = X.sample(n=sample_size, random_state=42)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(X_sample.isnull(), cbar=False, cmap="GnBu")
    plt.title(f"Visual Sparsity Map (Sample n={sample_size})")
    plt.savefig(output_dir / "sparsity_heatmap.png")
    plt.close()
    
    # Stats
    stats = {
        "n_features": int(n_cols),
        "n_rows": int(n_rows),
        "features_gt_80_missing": int((missing_by_feature > 0.80).sum()),
        "features_gt_50_missing": int((missing_by_feature > 0.50).sum()),
        "avg_missing_per_feature": float(missing_by_feature.mean()),
        "avg_missing_per_row": float(missing_by_row.mean())
    }
    return stats


def _analyze_metadata(X: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path, meta_dict: Dict) -> Dict:
    """
    Uses metadata to group features and analyze sparsity/coverage per group.
    """
    # Filter metadata to keep only features present in X
    valid_features = set(X.columns)
    meta_filtered = metadata[metadata['index'].isin(valid_features)].copy()
    
    if meta_filtered.empty:
        logger.warning("No matching features found in metadata!")
        return {}
        
    group_stats = {}
    
    # Group by 'Component' (e.g., Questionnaire, Lab)
    if 'Component' in meta_filtered.columns:
        # Sort groups by missingness for better visualization
        group_missing = []
        for group_name, group_df in meta_filtered.groupby('Component'):
            features = group_df['index'].tolist()
            features = [f for f in features if f in X.columns]
            if not features: continue
            
            avg_miss = X[features].isnull().mean().mean()
            group_missing.append({'Component': group_name, 'Avg Missing': avg_miss, 'Count': len(features)})
        
        gm_df = pd.DataFrame(group_missing).sort_values('Avg Missing', ascending=False)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=gm_df, x='Avg Missing', y='Component', hue='Component', palette="rocket", legend=False)
        plt.title("Average Missingness by Data Component")
        plt.xlabel("Avg Missing Rate (%)")
        plt.tight_layout()
        plt.savefig(output_dir / "missingness_by_component.png")
        plt.close()
        
        for group_name in gm_df['Component']:
            features_in_group = meta_filtered[meta_filtered['Component'] == group_name]['index'].tolist()
            features_in_group = [f for f in features_in_group if f in X.columns]
                
            subset = X[features_in_group]
            avg_missing = subset.isnull().mean().mean()
            
            group_stats[group_name] = {
                "n_features": len(features_in_group),
                "avg_missing": avg_missing
            }
            
            # Save a quick plot for this group's missingness
            plt.figure(figsize=(10, 6))
            sns.histplot(subset.isnull().mean() * 100, bins=20, kde=True, color='purple')
            plt.title(f"Missingness Distribution: {group_name}")
            plt.xlabel("Missing %")
            plt.savefig(output_dir / f"missing_hist_{group_name.replace(' ', '_')}.png")
            plt.close()

    return group_stats


def _analyze_target_conditional(X: pd.DataFrame, y: pd.Series, output_dir: Path, meta_dict: Dict):
    """
    Analyzes distribution of top features conditioned on target.
    Selects top 10 features with least missing values to plot.
    """
    missing_rate = X.isnull().mean().sort_values()
    top_features = missing_rate.head(10).index.tolist()
    
    for feat in top_features:
        try:
            plt.figure(figsize=(12, 7))
            desc = meta_dict.get(feat, feat)
            title = f"Distribution: {desc}"
            
            # Check if numeric or categorical (nunique is a good hint if type is unclear)
            if pd.api.types.is_numeric_dtype(X[feat]) and X[feat].nunique() > 10:
                sns.kdeplot(data=X, x=feat, hue=y, fill=True, common_norm=False, palette="viridis")
                plt.title(title)
                plt.xlabel(desc)
            else:
                # Top 10 categories if too many
                top_cats = X[feat].value_counts().nlargest(10).index
                sns.countplot(data=X[X[feat].isin(top_cats)], x=feat, hue=y, palette="coolwarm")
                plt.title(title)
                plt.xlabel(desc)
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.savefig(output_dir / f"dist_{feat}.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot feature {feat}: {e}")

def _analyze_filtering_potential(X: pd.DataFrame, output_dir: Path, meta_dict: Dict) -> Dict:
    """
    Identifies features likely to be filtered out and creates a justification plot.
    """
    filtering_dir = output_dir / "filtering_justification"
    filtering_dir.mkdir(parents=True, exist_ok=True)
    
    missing_rate = X.isnull().mean()
    high_missing = missing_rate[missing_rate > 0.8]
    
    variances = X.select_dtypes(include=[np.number]).var()
    low_variance = variances[variances <= 0.1]
    
    constant_features = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    
    reasons = {
        "High Missingness (>80%)": len(high_missing),
        "Low Variance (<=0.1)": len(low_variance),
        "Constant Value": len(constant_features)
    }
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(reasons.values()), y=list(reasons.keys()), hue=list(reasons.keys()), palette="flare", legend=False)
    plt.title("Potential Features to be Dropped by Reason")
    plt.xlabel("Number of Features")
    plt.savefig(filtering_dir / "potential_drops_summary.png")
    plt.close()
    
    # Save a detailed list of potential drops for justification
    details = []
    for f in high_missing.index:
        details.append({"feature": f, "label": meta_dict.get(f, f), "reason": "High Missingness", "value": f"{missing_rate[f]:.1%}"})
    for f in constant_features:
        if f not in high_missing.index:
            details.append({"feature": f, "label": meta_dict.get(f, f), "reason": "Constant Value", "value": "1 unique"})
            
    # Save top 20 justification to CSV or just return in report
    pd.DataFrame(details).head(50).to_csv(filtering_dir / "potential_drops_details.csv", index=False)
    
    return {
        "potential_drops_count": len(set(list(high_missing.index) + constant_features + list(low_variance.index))),
        "reasons_summary": reasons
    }

