
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path.cwd()))

from ml_pipeline.data.preprocessing import PreprocessingPipeline
from ml_pipeline.config import TARGET_COLUMN

# Mock data
data = {
    'AGQ130': [1, 2, np.nan, 4],
    'ALQ120Q': [np.nan, 1, 0, 1],
    'CAT_FEAT': ['A', 'B', 'A', np.nan],
    'y': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

pipeline = PreprocessingPipeline(target_col='y')
X = df.drop(columns=['y'])
y = df['y']

pipeline.fit(X, y)

print("Pipeline attribute 'pipeline' exists:", hasattr(pipeline, 'pipeline'))
if hasattr(pipeline, 'pipeline'):
    print("Feature names out:", pipeline.pipeline.get_feature_names_out())

# Check shape of transformed data
X_proc = pipeline.transform(X)
print("Transformed shape:", X_proc.shape)
