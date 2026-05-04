from __future__ import annotations

import pandas as pd

from fraud_mlops.data import temporal_split, validate_dataset
from fraud_mlops.encoding import TargetEncoder, identify_feature_columns
from fraud_mlops.monitoring import MonitoringThresholds, evaluate_alerts
from fraud_mlops.retraining import RetrainingPolicy, should_retrain


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [1, 2, 3, 4, 5, 6],
            "TransactionID": [11, 12, 13, 14, 15, 16],
            "isFraud": [0, 0, 0, 1, 0, 1],
            "num1": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "cat1": ["a", "b", "a", "c", None, "b"],
        }
    )


def test_validation_and_split() -> None:
    frame = sample_frame()
    report = validate_dataset(frame, "isFraud", "TransactionDT")
    train, test = temporal_split(frame, "TransactionDT", 0.33)

    assert report.row_count == 6
    assert report.missing_columns == []
    assert len(train) == 4
    assert len(test) == 2


def test_target_encoding_and_feature_detection() -> None:
    frame = sample_frame()
    train, _ = temporal_split(frame, "TransactionDT", 0.33)
    numeric_columns, categorical_columns = identify_feature_columns(train, "isFraud", "TransactionDT")
    encoder = TargetEncoder().fit(train, categorical_columns, "isFraud")
    encoded = encoder.transform(train.drop(columns=["isFraud"]), categorical_columns)

    assert "cat1_target_encoded" in encoded.columns
    assert numeric_columns == ["TransactionID", "num1"]


def test_retraining_and_alert_rules() -> None:
    policy = RetrainingPolicy(0.8, 0.2)
    alerts = evaluate_alerts(
        {"recall": 0.7, "false_positive_rate": 0.2, "latency_ms": 600, "drift_psi": 0.3, "missing_rate": 0.01},
        MonitoringThresholds(0.8, 0.1, 500, 0.2, 0.05),
    )

    assert should_retrain(0.7, 0.25, policy)
    assert "recall_drop" in alerts
    assert "latency_spike" in alerts
