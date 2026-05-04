from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_proba_array = np.asarray(y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array).ravel()
    false_positive_rate = fp / max(fp + tn, 1)

    return {
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true_array, y_proba_array)),
        "false_positive_rate": float(false_positive_rate),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }


def business_impact(y_true, y_pred, fraud_loss_per_case: float = 500.0, false_alarm_cost: float = 5.0) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fraud_loss = fn * fraud_loss_per_case
    false_alarm_cost_total = fp * false_alarm_cost
    return {
        "fraud_loss": float(fraud_loss),
        "false_alarm_cost": float(false_alarm_cost_total),
        "total_cost": float(fraud_loss + false_alarm_cost_total),
        "true_positives": float(tp),
        "false_negatives": float(fn),
        "false_positives": float(fp),
    }


def save_confusion_matrix_plot(y_true, y_pred, output_path: str | Path) -> Path:
    matrix = confusion_matrix(y_true, y_pred)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output
