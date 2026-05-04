from __future__ import annotations

import argparse

from fraud_mlops.config import TARGET_COLUMN, TIME_COLUMN
from fraud_mlops.data import load_dataset, missing_values_summary, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the IEEE dataset schema and missing values.")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = load_dataset(args.data)
    report = validate_dataset(df, TARGET_COLUMN, TIME_COLUMN)
    print(f"Rows: {report.row_count}")
    print(f"Duplicate rows: {report.duplicate_rows}")
    print(f"Missing required columns: {report.missing_columns}")
    print(missing_values_summary(df).head(20).to_string())


if __name__ == "__main__":
    main()
