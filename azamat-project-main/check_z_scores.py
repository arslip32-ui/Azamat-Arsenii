import pandas as pd
import numpy as np
from src.normalize import REFERENCE_RANGES, _SCORE_FEATURE_KEYS

# Load your processed readability scores
df = pd.read_csv('data/processed/readability_scores.csv')

# For each language, compute z-scores
for lang in ['en', 'ru', 'kk']:
    lang_data = df[df['lang'] == lang]
    
    print(f"\n{lang.upper()}:")
    for key in _SCORE_FEATURE_KEYS:
        if key in lang_data.columns:
            feature_values = lang_data[key].values
            mean = np.mean(feature_values)
            stdev = np.std(feature_values)
            
            # Normalize to z-scores
            z_scores = (feature_values - mean) / stdev
            
            print(f"  {key}: mean={mean:.2f}, stdev={stdev:.2f}, z_mean={np.mean(z_scores):.4f}")