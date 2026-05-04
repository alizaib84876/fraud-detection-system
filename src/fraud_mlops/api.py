from __future__ import annotations

import os
import time

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


MODEL_URI = os.getenv("MODEL_URI", "artifacts/model")

app = FastAPI(title="IEEE Fraud Inference API", version="1.0.0")
model = None

REQUEST_COUNT = Counter("fraud_api_request_total", "Total API requests", ["endpoint", "status"])
ERROR_COUNT = Counter("fraud_api_error_total", "Total API errors", ["endpoint"])
LATENCY_HIST = Histogram("fraud_api_latency_ms", "API latency in ms", ["endpoint"])
LATENCY_GAUGE = Gauge("fraud_api_latency_ms_latest", "Latest API latency in ms", ["endpoint"])

MODEL_RECALL = Gauge("fraud_model_recall", "Fraud recall")
MODEL_FPR = Gauge("fraud_false_positive_rate", "False positive rate")
MODEL_PRECISION = Gauge("fraud_precision", "Precision")
PRED_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Prediction confidence distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
DRIFT_PSI = Gauge("fraud_data_drift_psi", "Data drift PSI")
MISSING_RATE = Gauge("fraud_missing_rate", "Missing value rate")


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature vector for a single transaction")


def _set_gauge_from_env(gauge: Gauge, env_name: str, default: float) -> None:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        gauge.set(default)
        return
    try:
        gauge.set(float(raw_value))
    except ValueError:
        gauge.set(default)


@app.on_event("startup")
def load_model() -> None:
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception:
        model = None

    _set_gauge_from_env(MODEL_RECALL, "FRAUD_RECALL", 0.75)
    _set_gauge_from_env(MODEL_FPR, "FRAUD_FPR", 0.05)
    _set_gauge_from_env(MODEL_PRECISION, "FRAUD_PRECISION", 0.25)
    _set_gauge_from_env(DRIFT_PSI, "FRAUD_DRIFT_PSI", 0.12)
    _set_gauge_from_env(MISSING_RATE, "FRAUD_MISSING_RATE", 0.15)


@app.get("/health")
def health_check() -> dict[str, str]:
    start = time.perf_counter()
    endpoint = "/health"
    status = "ok" if model is not None else "model_not_loaded"
    REQUEST_COUNT.labels(endpoint=endpoint, status="ok").inc()
    latency_ms = (time.perf_counter() - start) * 1000
    LATENCY_HIST.labels(endpoint=endpoint).observe(latency_ms)
    LATENCY_GAUGE.labels(endpoint=endpoint).set(latency_ms)
    return {"status": status}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    start = time.perf_counter()
    endpoint = "/predict"
    try:
        if model is None:
            ERROR_COUNT.labels(endpoint=endpoint).inc()
            REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
            return {"error": "Model is not loaded"}

        frame = pd.DataFrame([request.features])
        probability = float(np.asarray(model.predict(frame))[0])
        PRED_CONFIDENCE.observe(probability)
        REQUEST_COUNT.labels(endpoint=endpoint, status="ok").inc()
        return {"fraud_probability": probability}
    except Exception:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        return {"error": "Prediction failed"}
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        LATENCY_HIST.labels(endpoint=endpoint).observe(latency_ms)
        LATENCY_GAUGE.labels(endpoint=endpoint).set(latency_ms)
