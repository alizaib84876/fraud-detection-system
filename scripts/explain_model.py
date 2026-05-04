from __future__ import annotations

import argparse

from fraud_mlops.config import TARGET_COLUMN, TIME_COLUMN
from fraud_mlops.data import load_dataset, temporal_split
from fraud_mlops.encoding import TargetEncoder, identify_feature_columns, impute_numeric
from fraud_mlops.explainability import extract_feature_importance, save_feature_importance_plot
from fraud_mlops.models import build_xgboost_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a feature-importance chart for the IEEE dataset.")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = load_dataset(args.data)
    train_df, test_df = temporal_split(df, TIME_COLUMN, 0.2)
    numeric_columns, categorical_columns = identify_feature_columns(train_df, TARGET_COLUMN, TIME_COLUMN)

    train_features = impute_numeric(train_df.drop(columns=[TARGET_COLUMN]), numeric_columns)
    test_features = impute_numeric(test_df.drop(columns=[TARGET_COLUMN]), numeric_columns)

    encoder = TargetEncoder()
    if categorical_columns:
        train_encoded = encoder.fit_transform(train_features.join(train_df[[TARGET_COLUMN]]), categorical_columns, TARGET_COLUMN).drop(columns=[TARGET_COLUMN])
        test_encoded = encoder.transform(test_features.fillna("__missing__"), categorical_columns)
    else:
        train_encoded = train_features
        test_encoded = test_features

    model = build_xgboost_model(random_state=42, scale_pos_weight=None)
    model.fit(train_encoded, train_df[TARGET_COLUMN].astype(int))
    importance = extract_feature_importance(model, list(train_encoded.columns))
    output = save_feature_importance_plot(importance, "reports/feature_importance.png")
    print(f"Saved feature importance to {output}")


if __name__ == "__main__":
    main()
