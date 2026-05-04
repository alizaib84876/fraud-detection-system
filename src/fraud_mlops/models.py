from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def build_xgboost_model(random_state: int, scale_pos_weight: float | None = None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight if scale_pos_weight and scale_pos_weight > 0 else 1.0,
        eval_metric="aucpr",
    )


def build_lightgbm_model(random_state: int, class_weight: str | dict | None = None) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )


def build_hybrid_rf_model(random_state: int, class_weight: str | dict | None = None) -> Pipeline:
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
        ),
        threshold="median",
    )
    classifier = RandomForestClassifier(
        n_estimators=350,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )
    return Pipeline([("feature_selection", selector), ("classifier", classifier)])
