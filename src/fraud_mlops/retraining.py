from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainingPolicy:
    recall_threshold: float
    drift_threshold: float
    mode: str = "hybrid"
    periodic_days: int = 7


def should_retrain(current_recall: float, drift_score: float, policy: RetrainingPolicy, days_since_last_train: int | None = None) -> bool:
    if policy.mode == "threshold":
        return current_recall < policy.recall_threshold or drift_score >= policy.drift_threshold
    if policy.mode == "periodic":
        return days_since_last_train is not None and days_since_last_train >= policy.periodic_days
    return (
        current_recall < policy.recall_threshold
        or drift_score >= policy.drift_threshold
        or (days_since_last_train is not None and days_since_last_train >= policy.periodic_days)
    )
