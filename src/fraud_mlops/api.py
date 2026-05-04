from __future__ import annotations

import os

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_URI = os.getenv("MODEL_URI", "artifacts/model")

app = FastAPI(title="IEEE Fraud Inference API", version="1.0.0")
model = None


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature vector for a single transaction")


@app.on_event("startup")
def load_model() -> None:
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception:
        model = None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok" if model is not None else "model_not_loaded"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if model is None:
        return {"error": "Model is not loaded"}
    frame = pd.DataFrame([request.features])
    probability = float(np.asarray(model.predict(frame))[0])
    return {"fraud_probability": probability}
