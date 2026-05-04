from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_time_based_drift(early_df: pd.DataFrame, later_df: pd.DataFrame, drift_strength: float = 0.15) -> pd.DataFrame:
    """Create a later-period sample with shifted fraud patterns.

    The function keeps the time order intact and makes the later period harder by
    perturbing numeric features and increasing the fraud rate on a small subset.
    """

    drifted = later_df.copy().reset_index(drop=True)
    if drifted.empty:
        return drifted

    numeric_columns = drifted.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = drifted.select_dtypes(exclude=[np.number]).columns.tolist()

    for column in numeric_columns[: min(10, len(numeric_columns))]:
        noise = np.random.normal(loc=0.0, scale=drift_strength * (drifted[column].std() or 1.0), size=len(drifted))
        drifted[column] = drifted[column].fillna(drifted[column].median()) + noise

    if categorical_columns:
        column = categorical_columns[0]
        drifted[column] = drifted[column].astype(str)
        drifted.loc[drifted.index[:: max(len(drifted) // 20, 1)], column] = "__new_fraud_pattern__"

    if "isFraud" in drifted.columns:
        fraud_indices = drifted[drifted["isFraud"] == 1].index
        if len(fraud_indices) > 0:
            extra_shift_count = max(int(len(fraud_indices) * drift_strength), 1)
            extra_indices = np.random.choice(fraud_indices, size=min(extra_shift_count, len(fraud_indices)), replace=False)
            drifted.loc[extra_indices, "isFraud"] = 1

    return drifted


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_values = pd.Series(expected).dropna()
    actual_values = pd.Series(actual).dropna()
    if expected_values.empty or actual_values.empty:
        return 0.0

    combined_bins = np.unique(np.quantile(expected_values, np.linspace(0, 1, bins + 1)))
    if len(combined_bins) < 3:
        return 0.0

    expected_hist, _ = np.histogram(expected_values, bins=combined_bins)
    actual_hist, _ = np.histogram(actual_values, bins=combined_bins)

    expected_pct = np.where(expected_hist == 0, 0.0001, expected_hist / expected_hist.sum())
    actual_pct = np.where(actual_hist == 0, 0.0001, actual_hist / actual_hist.sum())
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
