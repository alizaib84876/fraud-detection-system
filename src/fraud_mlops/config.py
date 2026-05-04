from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_REGISTRY_DIR = ARTIFACTS_DIR / "model_registry"

MLFLOW_EXPERIMENT_NAME = "ieee-fraud-mlflow"
MLFLOW_TRACKING_URI = f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"
ID_COLUMN = "TransactionID"

TEST_FRACTION = 0.2
RANDOM_STATE = 42
RECALL_THRESHOLD = 0.80
AUC_THRESHOLD = 0.90
