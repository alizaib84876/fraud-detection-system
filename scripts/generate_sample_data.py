from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_dataset(rows: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    fraud = rng.choice([0, 1], size=rows, p=[0.93, 0.07])

    frame = pd.DataFrame(
        {
            "TransactionDT": np.arange(1, rows + 1),
            "TransactionID": np.arange(1000000, 1000000 + rows),
            "isFraud": fraud,
            "num1": rng.normal(100, 25, size=rows),
            "num2": rng.normal(10, 4, size=rows),
            "cat1": rng.choice([f"merchant_{index}" for index in range(25)], size=rows),
            "cat2": rng.choice([f"card_{index}" for index in range(15)], size=rows),
        }
    )

    missing_index = rng.choice(frame.index, size=max(rows // 10, 1), replace=False)
    frame.loc[missing_index, "num1"] = np.nan
    frame.loc[missing_index[: max(len(missing_index) // 2, 1)], "cat1"] = None
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small IEEE-like fraud dataset for CI and local smoke tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_dataset(args.rows).to_csv(output, index=False)
    print(f"Wrote sample dataset to {output}")


if __name__ == "__main__":
    main()
