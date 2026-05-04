# Fraud Detection System (IEEE CIS)

End-to-end fraud detection pipeline using the IEEE CIS Fraud Detection dataset. The project focuses on high recall, scalable training, automated monitoring, and retraining triggers with MLflow for experiment tracking.

## Highlights

- MLflow experiment tracking with nested runs
- Data validation, preprocessing, and feature engineering
- High-cardinality categorical handling via target encoding
- Imbalance handling: standard, SMOTE, and undersampling
- Models: XGBoost, LightGBM, and hybrid RF + feature selection
- Metrics: precision, recall, F1, AUC-ROC, confusion matrix
- Cost-sensitive training comparison
- Drift simulation and retraining decision logic
- CI/CD with Docker images for training and inference
- Monitoring hooks for alerts and retraining triggers

## Dataset

Download the IEEE CIS Fraud Detection data from Kaggle and place the files in:

```
data/ieee-fraud-detection/
```

Required file for training:

- `train_transaction.csv`

The dataset files are excluded from Git by `.gitignore`.

## Quickstart

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Validate data:

```bash
python3 scripts/validate_data.py --data data/ieee-fraud-detection/train_transaction.csv
```

Run the MLflow pipeline:

```bash
python3 scripts/run_pipeline.py --data data/ieee-fraud-detection/train_transaction.csv
```

Start the MLflow UI:

```bash
mlflow ui
```

## Explainability

Generate a feature-importance plot:

```bash
python3 scripts/explain_model.py --data data/ieee-fraud-detection/train_transaction.csv
```

Output:

- `reports/feature_importance.png`

## Drift Simulation

```bash
python3 scripts/simulate_drift.py --data data/ieee-fraud-detection/train_transaction.csv
```

## CI/CD (GitHub Actions)

Workflow file:

- `.github/workflows/ci-cd.yml`

Builds and pushes Docker images to GHCR:

- `ghcr.io/<owner>/<repo>/ieee-fraud-train:latest`
- `ghcr.io/<owner>/<repo>/ieee-fraud-api:latest`

## Monitoring Trigger

Simulate monitoring-driven CI trigger (dry run):

```bash
python3 scripts/monitor_and_trigger.py --data data/ieee-fraud-detection/train_transaction.csv --dry-run
```

## Project Structure

```
src/fraud_mlops/        Core pipeline modules
scripts/               Entry points for training, validation, drift, and monitoring
docker/                Training + API Dockerfiles
monitoring/            Prometheus + Grafana assets
```
