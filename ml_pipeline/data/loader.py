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
        
        # Standardize columns? No, keep original casing for metadata matching.
        # df.columns = df.columns.str.lower()
        
        # Titanic specific renaming - removed for Hackathon
        # rename_map = { ... }
        # df = df.rename(columns=rename_map)
        
        # Generate dummy ID if missing? No, we expect strict ID column.
        # if "passengerid" not in df.columns: ...

            
        logger.info(f"Data loaded from {path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        raise
