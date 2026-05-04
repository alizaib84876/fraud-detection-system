# IEEE Fraud Detection MLOps Project

This repository is being adapted from the original Kubeflow wording to **MLflow** as requested by your instructor.

The goal is to build a fraud detection workflow for the IEEE CIS Fraud Detection dataset with:

- high recall for fraud cases
- scalable training and inference
- automated performance-degradation detection and retraining triggers

## What is already scaffolded

- MLflow-based experiment pipeline
- data validation and preprocessing helpers
- target encoding for high-cardinality categorical fields
- support for XGBoost, LightGBM, and a hybrid RF + feature-selection model
- imbalance handling hooks for class weighting and SMOTE
- evaluation helpers for precision, recall, F1, AUC-ROC, and confusion matrix
- assignment-ready screenshot checklist

## Screenshot checklist for proof

Use these screenshots as evidence in your submission.

| Step | What to capture | Suggested file or UI |
| --- | --- | --- |
| 1 | Environment verification | Terminal showing `python3 --version`, `docker --version`, `minikube version`, `kubectl version`, `mlflow --version` |
| 2 | Project scaffold | Workspace explorer showing this repository structure |
| 3 | Dependency installation | Terminal output from `pip install -r requirements.txt` |
| 4 | Data download | Kaggle dataset folder or CSV files in `data/raw/` |
| 5 | Data validation | Terminal output from the validation script and missing-value summary |
| 6 | MLflow tracking server | MLflow UI home page and experiment list |
| 7 | Pipeline run | Terminal output from the main pipeline command |
| 8 | Model comparison | MLflow run page showing XGBoost, LightGBM, and hybrid results |
| 9 | Imbalance strategy comparison | Metrics table or logs showing class weighting vs SMOTE |
| 10 | Cost-sensitive training | Metrics comparison between standard and cost-sensitive runs |
| 11 | Explainability | SHAP summary plot or feature-importance figure |
| 12 | Deployment decision | Logged conditional deployment artifact or approval flag |
| 13 | CI/CD run | GitHub Actions workflow run page |
| 14 | Monitoring dashboards | Grafana system/model/data drift dashboards |
| 15 | Alerting | Prometheus/Grafana alert firing and retraining trigger evidence |

## Suggested order for the assignment write-up

1. Environment setup
2. Dataset acquisition and schema check
3. MLflow experiment tracking setup
4. Data preprocessing and imbalance handling
5. Model training and evaluation
6. Cost-sensitive learning comparison
7. Drift simulation and retraining strategy
8. CI/CD and monitoring integration
9. Explainability and business impact analysis

## Local workflow

The current code is structured so you can run the pipeline from a single entry point once the dependencies and data are in place.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_pipeline.py --data data/raw/ieee_fraud_train_transaction.csv
```

If you use the MLflow UI locally, capture a screenshot of the experiment page after the pipeline completes.
