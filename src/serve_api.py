from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

MODELS_DIR = Path("models")

app = FastAPI(title="Threat Detection API", version="1.0.0")


class FlowRecord(BaseModel):
    # Example fields (not exhaustive)
    proto: Optional[str] = None
    service: Optional[str] = None
    state: Optional[str] = None
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    spkts: Optional[float] = None
    dpkts: Optional[float] = None
    sbytes: Optional[float] = None
    dbytes: Optional[float] = None


class PredictRequest(BaseModel):
    model_name: str = "random_forest"
    records: List[FlowRecord]
    # Silence protected namespace warning for field name "model_name"
    model_config = {"protected_namespaces": ()}


class PredictResponse(BaseModel):
    prediction: List[int]
    threat_probability: Optional[List[float]] = None


@app.on_event("startup")
def init_cache():
    app.state.preprocessor_bundle = None
    app.state.models_cache = {}


def ensure_preprocessor_loaded():
    if getattr(app.state, "preprocessor_bundle", None) is None:
        try:
            app.state.preprocessor_bundle = joblib.load(MODELS_DIR / "preprocessor.joblib")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Preprocessor not available. Run preprocessing/training first.") from exc
    return app.state.preprocessor_bundle


def get_model(model_name: str):
    if model_name not in app.state.models_cache:
        try:
            app.state.models_cache[model_name] = joblib.load(MODELS_DIR / f"model_{model_name}.joblib")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' not found. Train the model first.") from exc
    return app.state.models_cache[model_name]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model(req.model_name)
    preprocessor_bundle = ensure_preprocessor_loaded()
    preprocessor = preprocessor_bundle["preprocessor"]

    df = pd.DataFrame([r.model_dump() for r in req.records])
    X = preprocessor.transform(df)
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1].tolist()

    return PredictResponse(prediction=[int(p) for p in preds.tolist()], threat_probability=probs) 