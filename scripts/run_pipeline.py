from __future__ import annotations

import argparse

from fraud_mlops.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IEEE fraud MLflow pipeline.")
    parser.add_argument("--data", required=True, help="Path to the IEEE training transaction CSV")
    args = parser.parse_args()

    results = run_pipeline(args.data)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
