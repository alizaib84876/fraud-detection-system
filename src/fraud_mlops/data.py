from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    missing_columns: list[str]
    duplicate_rows: int
    row_count: int
    missing_summary: pd.DataFrame


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def validate_dataset(df: pd.DataFrame, target_column: str, time_column: str) -> ValidationReport:
    missing_columns = [column for column in [target_column, time_column] if column not in df.columns]
    duplicate_rows = int(df.duplicated().sum())
    missing_summary = missing_values_summary(df)
    return ValidationReport(
        missing_columns=missing_columns,
        duplicate_rows=duplicate_rows,
        row_count=int(len(df)),
        missing_summary=missing_summary,
    )


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({"missing_count": df.isna().sum()})
    summary["missing_rate"] = summary["missing_count"] / max(len(df), 1)
    summary = summary[summary["missing_count"] > 0]
    return summary.sort_values("missing_rate", ascending=False)


def temporal_split(df: pd.DataFrame, time_column: str, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(time_column).reset_index(drop=True)
    split_index = max(int(len(ordered) * (1 - test_fraction)), 1)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()
