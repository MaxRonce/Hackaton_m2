import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ..config import EDA_DIR, TARGET_COLUMN
from ..utils.logger import setup_logger

logger = setup_logger("data.analysis")

class AutoEDA:
    def __init__(self, data: pd.DataFrame, target_col: str = TARGET_COLUMN):
        self.data = data
        self.target_col = target_col
        self.output_dir = EDA_DIR

    def run(self):
        """Run all EDA steps."""
        logger.info("Starting Automated EDA...")
        self.plot_target_distribution()
        self.plot_numerical_features()
        self.plot_categorical_features()
        self.plot_correlation_matrix()
        self.plot_missing_values()
        logger.info(f"EDA completed. Plots saved to {self.output_dir}")

    def plot_target_distribution(self):
        if self.target_col not in self.data.columns:
            logger.warning(f"Target column {self.target_col} not found. Skipping target plot.")
            return

        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.target_col, data=self.data)
        plt.title("Target Distribution")
        plt.savefig(self.output_dir / "target_distribution.png")
        plt.close()

    def plot_numerical_features(self):
        num_cols = self.data.select_dtypes(include=['number']).columns
        if self.target_col in num_cols:
            num_cols = num_cols.drop(self.target_col)
        
        for col in num_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.data, x=col, kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(self.output_dir / f"numeric_{col}_dist.png")
            plt.close()

    def plot_categorical_features(self):
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            # Skip high cardinality
            if self.data[col].nunique() > 50:
                continue
                
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.data, x=col)
            plt.title(f"Count of {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"cat_{col}_counts.png")
            plt.close()

    def plot_correlation_matrix(self):
        num_cols = self.data.select_dtypes(include=['number'])
        if num_cols.empty:
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png")
        plt.close()

    def plot_missing_values(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.savefig(self.output_dir / "missing_values.png")
        plt.close()
