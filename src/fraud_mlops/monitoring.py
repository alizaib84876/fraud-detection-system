from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitoringThresholds:
    recall_min: float
    fpr_max: float
    latency_ms_max: float
    drift_psi_max: float
    missing_rate_max: float


def evaluate_alerts(metrics: dict[str, float], thresholds: MonitoringThresholds) -> list[str]:
    alerts: list[str] = []
    if metrics.get("recall", 1.0) < thresholds.recall_min:
        alerts.append("recall_drop")
    if metrics.get("false_positive_rate", 0.0) > thresholds.fpr_max:
        alerts.append("false_positive_rate_high")
    if metrics.get("latency_ms", 0.0) > thresholds.latency_ms_max:
        alerts.append("latency_spike")
    if metrics.get("drift_psi", 0.0) > thresholds.drift_psi_max:
        alerts.append("data_drift")
    if metrics.get("missing_rate", 0.0) > thresholds.missing_rate_max:
        alerts.append("missingness_spike")
    return alerts
