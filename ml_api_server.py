from pathlib import Path
from typing import Optional
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("IRRIGATION_MODEL_PATH", "irrigation_model_scan_hourly.pkl"))


class PredictionRequest(BaseModel):
    moisture: float = Field(..., description="Soil moisture reading.")
    temperature: float = Field(..., description="Soil temperature reading.")
    humidity: Optional[float] = Field(default=None, description="Optional air humidity reading.")
    soil_ph: Optional[float] = Field(default=None, description="Optional soil pH reading.")
    zone: Optional[str] = Field(default=None, description="Optional field zone identifier.")


class PredictionResponse(BaseModel):
    recommendation: str
    confidence: dict[str, float]
    features_used: list[str]


app = FastAPI(
    title="Irrigation Recommendation API",
    description="Predict irrigation actions from soil sensor readings.",
    version="1.0.0",
)


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_path": str(MODEL_PATH.resolve())}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    row = {
        "moisture": request.moisture,
        "temperature": request.temperature,
        "humidity": request.humidity,
        "soil_ph": request.soil_ph,
        "zone": request.zone,
    }
    input_df = pd.DataFrame([row]).reindex(columns=feature_columns)

    recommendation = model.predict(input_df)[0]

    confidence: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        labels = model.named_steps["classifier"].classes_
        confidence = {
            str(label): round(float(score), 4)
            for label, score in sorted(
                zip(labels, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    return PredictionResponse(
        recommendation=str(recommendation),
        confidence=confidence,
        features_used=feature_columns,
    )
