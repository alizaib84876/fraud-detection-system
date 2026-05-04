from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TargetEncoder:
    smoothing: float = 20.0
    global_mean: float = 0.0
    mapping_: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: list[str], target: str) -> "TargetEncoder":
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found")

        self.global_mean = float(df[target].mean())
        self.mapping_ = {}

        for column in columns:
            grouped = df.groupby(column, dropna=False)[target].agg(["mean", "count"])
            smoothing = 1 / (1 + np.exp(-(grouped["count"] - self.smoothing) / self.smoothing))
            encoded = self.global_mean * (1 - smoothing) + grouped["mean"] * smoothing
            self.mapping_[column] = encoded.to_dict()

        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        transformed = df.copy()
        for column in columns:
            encoded_name = f"{column}_target_encoded"
            mapping = self.mapping_.get(column, {})
            transformed[encoded_name] = transformed[column].map(mapping).fillna(self.global_mean)
        return transformed.drop(columns=columns)

    def fit_transform(self, df: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
        return self.fit(df, columns, target).transform(df, columns)


def identify_feature_columns(df: pd.DataFrame, target_column: str, time_column: str | None = None) -> tuple[list[str], list[str]]:
    excluded = {target_column}
    if time_column:
        excluded.add(time_column)

    categorical_columns = [column for column in df.columns if column not in excluded and df[column].dtype == "object"]
    numeric_columns = [column for column in df.columns if column not in excluded and column not in categorical_columns]
    return numeric_columns, categorical_columns


def impute_numeric(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    filled = df.copy()
    for column in numeric_columns:
        median_value = filled[column].median()
        filled[column] = filled[column].fillna(median_value)
    return filled
