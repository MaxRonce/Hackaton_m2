
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

def run_advanced_eda(data_path: Path, metadata_path: Path, output_dir: Path):
    """
    Runs advanced EDA including sparsity analysis, metadata-driven analysis,
    and target-conditional distributions.
    """
    logger.info("Starting Advanced EDA...")
    
    # Load Data
    data = pd.read_csv(data_path)
    metadata = pd.read_csv(metadata_path)
    
    
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
    summary_report["sparsity"] = _analyze_sparsity(X, sparsity_dir)

    # 2. Metadata Analysis
    logger.info("Running Metadata Analysis...")
    summary_report["metadata_groups"] = _analyze_metadata(X, metadata, groups_dir)

    # 3. Target Conditional Analysis
    if y is not None:
        logger.info("Running Target Conditional Analysis...")
        _analyze_target_conditional(X, y, target_dir)
        
    # Save Summary Report
    with open(output_dir / "summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=4)
        
    logger.info(f"Advanced EDA completed. Results saved to {output_dir}")


def _analyze_sparsity(X: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Analyzes missing values: % per feature, % per row, histograms, heatmap.
    """
    n_rows, n_cols = X.shape
    
    # Feature Missingness
    missing_by_feature = X.isnull().mean()
    missing_by_feature_count = X.isnull().sum()
    
    plt.figure(figsize=(10, 6))
    (missing_by_feature * 100).hist(bins=50)
    plt.title("Histogram of Missing Values % per Feature")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Count of Features")
    plt.savefig(output_dir / "missing_per_feature_hist.png")
    plt.close()
    
    # Row Missingness
    missing_by_row = X.isnull().mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    (missing_by_row * 100).hist(bins=50)
    plt.title("Histogram of Missing Values % per Individual")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Count of Individuals")
    plt.savefig(output_dir / "missing_per_row_hist.png")
    plt.close()
    
    # Sparsity Heatmap (sampled if too large)
    sample_size = min(1000, n_rows)
    X_sample = X.sample(n=sample_size, random_state=42)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(X_sample.isnull(), cbar=False, cmap="viridis")
    plt.title(f"Sparsity Heatmap (Sample n={sample_size})")
    plt.savefig(output_dir / "sparsity_heatmap.png")
    plt.close()
    
    # Stats
    stats = {
        "n_features": int(n_cols),
        "n_rows": int(n_rows),
        "features_gt_95_missing": int((missing_by_feature > 0.95).sum()),
        "features_gt_50_missing": int((missing_by_feature > 0.50).sum()),
        "avg_missing_per_feature": float(missing_by_feature.mean()),
        "avg_missing_per_row": float(missing_by_row.mean())
    }
    return stats


def _analyze_metadata(X: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Uses metadata to group features and analyze sparsity/coverage per group.
    """
    # Merge column index with metadata to ensure we only look at existing columns
    # Assumes 'index' in metadata corresponds to column names or there is a column 'index'
    # Based on previous `head`, metadata has `index` col.
    
    # Check if 'index' is identifying column
    if 'index' not in metadata.columns:
        # Try to infer or fallback. The user showed "index,SAS,Component..."
        pass
        
    # Filter metadata to keep only features present in X
    valid_features = set(X.columns)
    meta_filtered = metadata[metadata['index'].isin(valid_features)].copy()
    
    if meta_filtered.empty:
        logger.warning("No matching features found in metadata!")
        return {}
        
    group_stats = {}
    
    # Group by 'Component' (e.g., Questionnaire, Lab)
    if 'Component' in meta_filtered.columns:
        for group_name, group_df in meta_filtered.groupby('Component'):
            features_in_group = group_df['index'].tolist()
            # Intersection with X
            features_in_group = [f for f in features_in_group if f in X.columns]
            
            if not features_in_group:
                continue
                
            subset = X[features_in_group]
            avg_missing = subset.isnull().mean().mean()
            
            group_stats[group_name] = {
                "n_features": len(features_in_group),
                "avg_missing": avg_missing
            }
            
            # Save a quick plot for this group's missingness
            plt.figure(figsize=(8, 4))
            subset.isnull().mean().hist()
            plt.title(f"Missingness % for {group_name}")
            plt.savefig(output_dir / f"missing_hist_{group_name}.png")
            plt.close()

    return group_stats


def _analyze_target_conditional(X: pd.DataFrame, y: pd.Series, output_dir: Path):
    """
    Analyzes distribution of top features conditioned on target.
    Selects top 10 features with least missing values to plot.
    """
    missing_rate = X.isnull().mean().sort_values()
    top_features = missing_rate.head(10).index.tolist()
    
    for feat in top_features:
        try:
            plt.figure(figsize=(8, 5))
            # Check if numeric or categorical
            if pd.api.types.is_numeric_dtype(X[feat]):
                sns.kdeplot(data=X, x=feat, hue=y, fill=True, common_norm=False)
                plt.title(f"Distribution of {feat} by Target")
            else:
                # Top 10 categories if too many
                top_cats = X[feat].value_counts().nlargest(10).index
                sns.countplot(data=X[X[feat].isin(top_cats)], x=feat, hue=y)
                plt.title(f"Count of {feat} by Target")
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.savefig(output_dir / f"dist_{feat}.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot feature {feat}: {e}")

