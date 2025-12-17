from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
from typing import Tuple, Optional
from ..utils.io import save_pickle, load_pickle
from ..utils.logger import setup_logger
from ..config import CACHE_DIR

logger = setup_logger("utils.cache")

def get_cache_path(data_path: str | Path, suffix: str) -> Path:
    """Generate a cache path based on the input file hash."""
    path = Path(data_path)
    # Simple hash of filename and modificaton time for speed
    # ideally content hash but for large files modtime is faster proxy
    stat = path.stat()
    identifier = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
    hash_str = hashlib.md5(identifier.encode()).hexdigest()
    return CACHE_DIR / f"{hash_str}_{suffix}.pkl"

def load_from_cache(data_path: str | Path, suffix: str) -> Optional[object]:
    cache_path = get_cache_path(data_path, suffix)
    if cache_path.exists():
        logger.info(f"Loading {suffix} from cache: {cache_path}")
        return load_pickle(cache_path)
    return None

def save_to_cache(obj: object, data_path: str | Path, suffix: str):
    cache_path = get_cache_path(data_path, suffix)
    save_pickle(obj, cache_path)
    logger.info(f"Saved {suffix} to cache: {cache_path}")
