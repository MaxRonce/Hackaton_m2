import os
from pathlib import Path

# --- Global Configurations ---
SEED = 42

# -- Export configuration --
GROUP_TOKEN = 42

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
MODELS_DIR = OUTPUTS_DIR / "models"
CACHE_DIR = OUTPUTS_DIR / "cache"

# Ensure directories exist
for directory in [DATA_DIR, OUTPUTS_DIR, EDA_DIR, MODELS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Data Config ---
TARGET_COLUMN = "y"
ID_COLUMN = "SEQN"
DROP_COLUMNS = []

# --- Metadata & Test Config ---
METADATA_PATH = DATA_DIR / "features_metadata.csv"
TEST_INDEXES_PATH = DATA_DIR / "test_indexes.csv"

