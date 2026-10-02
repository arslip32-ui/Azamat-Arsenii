import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

_SCORE_FEATURE_KEYS = [
    "char_density",
    "avg_syllables",
    "pos_ratio",
    "tree_depth",
]


def optimize_ranges_and_weights(
    df: pd.DataFrame,
    feature_keys: List[str] = _SCORE_FEATURE_KEYS,
    languages: List[str] = ["en", "ru", "kk"],
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    target_col: str = None,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[str, Dict[str, float]]]:
    """Calculates empirical reference ranges and normalized weights per language.

    Args:
        df: DataFrame containing feature columns and language column ('lang').
        feature_keys: List of feature names used for scoring.
        languages: Language codes to process.
        low_percentile: Lower percentile bound for range minimums.
        high_percentile: Upper percentile bound for range maximums.
        target_col: Optional column name for target grade levels to fit weights
          via regression.

    Returns:
        Tuple containing (REFERENCE_RANGES, LANGUAGE_WEIGHTS)
    """
    reference_ranges = {}
    language_weights = {}

    for lang in languages:
        lang_data = df[df["lang"] == lang]
        if lang_data.empty:
            continue

        reference_ranges[lang] = {}
        feature_stds = {}

        # 1. Compute empirical percentile bounds for REFERENCE_RANGES
        for key in feature_keys:
            if key in lang_data.columns:
                values = lang_data[key].dropna().values
                if len(values) == 0:
                    continue

                p_min = np.percentile(values, low_percentile)
                p_max = np.percentile(values, high_percentile)

                # Ensure minimum non-zero range width
                if p_min >= p_max:
                    p_min = np.min(values)
                    p_max = (
                        np.max(values)
                        if np.max(values) > p_min
                        else p_min + 1.0
                    )

                reference_ranges[lang][key] = (
                    round(float(p_min), 3),
                    round(float(p_max), 3),
                )
                feature_stds[key] = np.std(values)

        # 2. Compute normalized weights
        weights = {}
        if target_col and target_col in lang_data.columns:
            # Fit weights against target grade levels using Ridge Regression
            from sklearn.linear_model import Ridge

            valid_data = lang_data[feature_keys + [target_col]].dropna()
            X = valid_data[feature_keys]
            y = valid_data[target_col]

            # Standardize features to compare coefficient magnitudes
            X_std = (X - X.mean()) / (X.std() + 1e-8)
            model = Ridge(alpha=1.0)
            model.fit(X_std, y)

            abs_coefs = np.abs(model.coef_)
            norm_weights = abs_coefs / abs_coefs.sum()
            for key, w in zip(feature_keys, norm_weights):
                weights[key] = round(float(w), 3)
        else:
            # Default: Equal weighting normalized to sum to 1.0
            num_features = len(reference_ranges[lang])
            equal_weight = (
                round(1.0 / num_features, 3) if num_features > 0 else 0.25
            )
            weights = {key: equal_weight for key in reference_ranges[lang].keys()}

            # Adjust last weight so the sum is exactly 1.0
            total = sum(weights.values())
            if total != 1.0 and len(weights) > 0:
                first_key = list(weights.keys())[0]
                weights[first_key] = round(
                    weights[first_key] + (1.0 - total), 3
                )

        language_weights[lang] = weights

    return reference_ranges, language_weights


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/processed/readability_scores.csv")

    # Run optimization
    ranges, weights = optimize_ranges_and_weights(df)

    # Print output formatted for src/normalize.py
    print("REFERENCE_RANGES = {")
    for lang, r_dict in ranges.items():
        print(f"    '{lang}': {{")
        for f_key, (r_min, r_max) in r_dict.items():
            print(f"        '{f_key}': ({r_min}, {r_max}),")
        print("    },")
    print("}\n")

    print("LANGUAGE_WEIGHTS = {")
    for lang, w_dict in weights.items():
        print(f"    '{lang}': {w_dict},")
    print("}")
