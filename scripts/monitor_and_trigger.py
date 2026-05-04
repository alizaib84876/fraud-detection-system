from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import mlflow
import requests

from fraud_mlops.config import MLFLOW_EXPERIMENT_NAME, RECALL_THRESHOLD
from fraud_mlops.drift import population_stability_index
from fraud_mlops.data import load_dataset, temporal_split


def get_latest_recall(experiment_name: str) -> Optional[float]:
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment not found: {experiment_name}")
        return None
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=50)
    for r in runs:
        if "recall" in r.data.metrics:
            return float(r.data.metrics["recall"])
    return None


def compute_drift_score(data_path: str, feature: str = "num1") -> float:
    df = load_dataset(data_path)
    early, later = temporal_split(df, "TransactionDT", 0.2)
    if feature not in early.columns or feature not in later.columns:
        return 0.0
    return population_stability_index(early[feature], later[feature])


def trigger_github_dispatch(repo: str, event_type: str, token: str, client_payload: dict | None = None) -> bool:
    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}
    payload = {"event_type": event_type}
    if client_payload:
        payload["client_payload"] = client_payload
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    if resp.status_code in (204, 201):
        return True
    print("GitHub dispatch failed:", resp.status_code, resp.text)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor MLflow metrics and data drift and trigger CI via GitHub repository_dispatch.")
    parser.add_argument("--data", required=True, help="Path to transaction CSV")
    parser.add_argument("--repo", required=False, help="GitHub repo owner/repo for dispatch (e.g., user/repo)")
    parser.add_argument("--token", required=False, help="GitHub token with repo:dispatch scope (or set GITHUB_TOKEN env)")
    parser.add_argument("--feature", default="num1", help="Numeric feature to compute PSI on")
    parser.add_argument("--drift-threshold", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recall = get_latest_recall(MLFLOW_EXPERIMENT_NAME)
    drift = compute_drift_score(args.data, args.feature)
    print(f"Latest recall: {recall}")
    print(f"Drift (PSI) on {args.feature}: {drift}")

    should_alert_recall = recall is not None and recall < RECALL_THRESHOLD
    should_alert_drift = drift >= args.drift_threshold

    if not should_alert_recall and not should_alert_drift:
        print("No alerts triggered.")
        return

    event_type = "model_drift_alert" if should_alert_drift else "recall_drop_alert"
    payload = {"recall": recall, "drift": drift}

    token = args.token or os.getenv("GITHUB_TOKEN")
    if args.dry_run or not token or not args.repo:
        print("DRY RUN - would dispatch:", event_type, "to repo", args.repo)
        print("Payload:", payload)
        if not token:
            print("No GITHUB_TOKEN provided; set env GITHUB_TOKEN or pass --token to actually dispatch.")
        return

    ok = trigger_github_dispatch(args.repo, event_type, token, client_payload=payload)
    print("Dispatched:" if ok else "Dispatch failed")


if __name__ == "__main__":
    main()
