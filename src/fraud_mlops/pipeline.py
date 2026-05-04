from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fraud_mlops.config import (
    AUC_THRESHOLD,
    ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
    RECALL_THRESHOLD,
    TARGET_COLUMN,
    TEST_FRACTION,
    TIME_COLUMN,
)
from fraud_mlops.data import load_dataset, missing_values_summary, temporal_split, validate_dataset
from fraud_mlops.encoding import TargetEncoder, identify_feature_columns, impute_numeric
from fraud_mlops.evaluation import business_impact, compute_metrics, save_confusion_matrix_plot
from fraud_mlops.models import build_hybrid_rf_model, build_lightgbm_model, build_xgboost_model


def _prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns, categorical_columns = identify_feature_columns(train_df, target_column, TIME_COLUMN)

    train_features = train_df.drop(columns=[target_column]).copy()
    test_features = test_df.drop(columns=[target_column]).copy()

    train_features = impute_numeric(train_features, numeric_columns)
    test_features = impute_numeric(test_features, numeric_columns)

    if categorical_columns:
        encoder = TargetEncoder()
        train_encoded = encoder.fit_transform(pd.concat([train_features, train_df[[target_column]]], axis=1), categorical_columns, target_column)
        train_encoded = train_encoded.drop(columns=[target_column])
        test_seed = test_features.copy()
        for column in categorical_columns:
            test_seed[column] = test_seed[column].fillna("__missing__")
        test_encoded = encoder.transform(test_seed, categorical_columns)
    else:
        train_encoded = train_features
        test_encoded = test_features

    return train_encoded, test_encoded


def _resample_if_needed(x_train: pd.DataFrame, y_train: pd.Series, strategy: str) -> tuple[pd.DataFrame, pd.Series]:
    if strategy != "smote":
        if strategy != "undersample":
            return x_train, y_train

        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        resampled_x, resampled_y = sampler.fit_resample(x_train, y_train)
        return pd.DataFrame(resampled_x, columns=x_train.columns), pd.Series(resampled_y)

    minority_count = int(y_train.value_counts().min())
    if minority_count <= 1:
        return x_train, y_train

    sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, minority_count - 1))
    resampled_x, resampled_y = sampler.fit_resample(x_train, y_train)
    return pd.DataFrame(resampled_x, columns=x_train.columns), pd.Series(resampled_y)


def _build_model(model_name: str, y_train: pd.Series, cost_sensitive: bool):
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / max(positives, 1) if cost_sensitive else 1.0
    class_weight = "balanced" if cost_sensitive else None

    if model_name == "xgboost":
        return build_xgboost_model(RANDOM_STATE, scale_pos_weight=scale_pos_weight)
    if model_name == "lightgbm":
        return build_lightgbm_model(RANDOM_STATE, class_weight=class_weight)
    if model_name == "hybrid_rf":
        return build_hybrid_rf_model(RANDOM_STATE, class_weight=class_weight)
    raise ValueError(f"Unknown model: {model_name}")


def run_pipeline(data_path: str | Path) -> pd.DataFrame:
    mlruns_path = Path(MLFLOW_TRACKING_URI.replace("file:", ""))
    mlruns_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_TRACKING_URI,
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_dataset(data_path)
    validation_report = validate_dataset(df, TARGET_COLUMN, TIME_COLUMN)
    missing_summary = missing_values_summary(df)

    if validation_report.missing_columns:
        raise ValueError(f"Missing required columns: {validation_report.missing_columns}")

    train_df, test_df = temporal_split(df, TIME_COLUMN, TEST_FRACTION)
    x_train, x_test = _prepare_features(train_df, test_df, TARGET_COLUMN)
    y_train = train_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    results: list[dict[str, float | str]] = []

    with mlflow.start_run(run_name="ieee_fraud_experiments"):
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("missing_feature_count", int(len(missing_summary)))
        mlflow.log_param("duplicate_rows", validation_report.duplicate_rows)

        for imbalance_strategy in ["standard", "smote", "undersample"]:
            resampled_x, resampled_y = _resample_if_needed(x_train, y_train, imbalance_strategy)
            for training_mode in ["standard", "cost_sensitive"]:
                for model_name in ["xgboost", "lightgbm", "hybrid_rf"]:
                    with mlflow.start_run(run_name=f"{model_name}_{imbalance_strategy}_{training_mode}", nested=True):
                        model = _build_model(model_name, resampled_y, cost_sensitive=training_mode == "cost_sensitive")
                        model.fit(resampled_x, resampled_y)

                        y_pred = model.predict(x_test)
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(x_test)[:, 1]
                        else:
                            y_proba = np.asarray(model.predict(x_test), dtype=float)

                        metrics = compute_metrics(y_test, y_pred, y_proba)
                        impact = business_impact(y_test, y_pred)

                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(metric_name, metric_value)
                        for impact_name, impact_value in impact.items():
                            mlflow.log_metric(impact_name, impact_value)

                        confusion_path = save_confusion_matrix_plot(
                            y_test,
                            y_pred,
                            ARTIFACTS_DIR / f"confusion_{model_name}_{imbalance_strategy}_{training_mode}.png",
                        )
                        mlflow.log_artifact(str(confusion_path))

                        results.append(
                            {
                                "model": model_name,
                                "imbalance_strategy": imbalance_strategy,
                                "training_mode": training_mode,
                                **metrics,
                                **impact,
                            }
                        )

                        if metrics["recall"] >= RECALL_THRESHOLD and metrics["auc_roc"] >= AUC_THRESHOLD:
                            approved_path = ARTIFACTS_DIR / "deployment" / f"approved_{model_name}_{imbalance_strategy}_{training_mode}.txt"
                            approved_path.parent.mkdir(parents=True, exist_ok=True)
                            approved_path.write_text(
                                f"approved=true\nmodel={model_name}\nimbalance_strategy={imbalance_strategy}\ntraining_mode={training_mode}\nrecall={metrics['recall']}\nauc_roc={metrics['auc_roc']}\n",
                                encoding="utf-8",
                            )
                            mlflow.log_artifact(str(approved_path))
                            mlflow.sklearn.log_model(model, artifact_path="model")

    return pd.DataFrame(results).sort_values(["recall", "auc_roc"], ascending=False)
