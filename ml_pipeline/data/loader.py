import pandas as pd
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger("data.loader")

def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        path (str | Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        df = pd.read_csv(path)
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        rename_map = {
            "siblings/spouses aboard": "sibsp",
            "parents/children aboard": "parch"
        }
        df = df.rename(columns=rename_map)
        
        # Generate dummy ID if missing (for compatibility)
        if "passengerid" not in df.columns:
            df["passengerid"] = df.index + 1
            
        logger.info(f"Data loaded from {path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        raise
