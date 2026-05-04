from __future__ import annotations

import argparse

from fraud_mlops.data import load_dataset, temporal_split
from fraud_mlops.drift import simulate_time_based_drift


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate time-based drift in the IEEE dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--time-column", default="TransactionDT")
    args = parser.parse_args()

    df = load_dataset(args.data)
    early, later = temporal_split(df, args.time_column, 0.2)
    drifted = simulate_time_based_drift(early, later)
    print(f"Early rows: {len(early)}")
    print(f"Later rows: {len(later)}")
    print(f"Drifted rows: {len(drifted)}")


if __name__ == "__main__":
    main()
