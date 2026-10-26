"""FastAPI service exposing Glimpse predictions."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from glimpse.explain.shap_explainer import explain_sample
from glimpse.models.trainer import load_model

app = FastAPI(title="Glimpse API", version="0.1.0")

_model = None
_features = None


def _ensure_loaded():
    global _model, _features
    if _model is None:
        try:
            _model, _features = load_model()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python -m glimpse.cli train` first.",
            ) from exc


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Ordered vector matching the trained feature list")


class Contribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictResponse(BaseModel):
    risk_score: float
    risk_label: str
    top_contributions: list[Contribution]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/features")
def features() -> dict:
    _ensure_loaded()
    return {"n_features": len(_features), "features": _features}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    _ensure_loaded()
    x = np.asarray(req.features, dtype=float)
    if x.shape[0] != len(_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(_features)} features, got {x.shape[0]}",
        )
    result = explain_sample(_model, _features, x)
    return PredictResponse(**result)