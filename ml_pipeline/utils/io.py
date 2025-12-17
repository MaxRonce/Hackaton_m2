import joblib
import json
import pandas as pd
from pathlib import Path
from .logger import setup_logger

logger = setup_logger("utils.io")

def save_pickle(obj, path):
    """Save object to pickle file (using joblib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as f:
            joblib.dump(obj, f)
        logger.info(f"Object saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {path}: {e}")
        raise

def load_pickle(path):
    """Load object from pickle file (using joblib)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, "rb") as f:
            obj = joblib.load(f)
        logger.info(f"Object loaded from {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load pickle from {path}: {e}")
        raise

def save_json(data, path):
    """Save dict to json file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise

def load_json(path):
    """Load dict from json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"JSON loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise
