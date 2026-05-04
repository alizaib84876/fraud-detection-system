from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_feature_importance(model, feature_names: list[str]) -> list[tuple[str, float]]:
    estimator = getattr(model, "named_steps", {}).get("classifier", model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []
    pairs = list(zip(feature_names, np.asarray(importances, dtype=float)))
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def save_feature_importance_plot(importance_pairs: list[tuple[str, float]], output_path: str | Path, top_n: int = 20) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    top_pairs = importance_pairs[:top_n]
    if not top_pairs:
        raise ValueError("No feature importances available")

    labels = [name for name, _ in top_pairs][::-1]
    values = [score for _, score in top_pairs][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
    ax.barh(labels, values, color="#2c7fb8")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def save_shap_summary(model, x_sample, output_path: str | Path) -> Path:
    import shap

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    estimator = getattr(model, "named_steps", {}).get("classifier", model)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(x_sample)
    shap.summary_plot(shap_values, x_sample, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    return output
