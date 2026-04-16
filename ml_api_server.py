from pathlib import Path
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from prediction_log_payload import build_prediction_log_record
from prediction_response_utils import (
    build_class_confidence_map,
    build_prediction_payload,
    validate_prediction_input,
)


MODEL_PATH = Path(os.getenv("IRRIGATION_MODEL_PATH", "irrigation_model_scan_hourly.pkl"))
MODEL_VERSION = os.getenv("IRRIGATION_MODEL_VERSION", "scan_hourly_v1")


class PredictionRequest(BaseModel):
    moisture: float = Field(..., description="Soil moisture reading.")
    temperature: float = Field(..., description="Soil temperature reading.")
    humidity: float = Field(..., description="Air humidity reading.")
    zone: str = Field(..., description="Field zone identifier.")


class PredictionResponse(BaseModel):
    recommendation: str | None
    confidence_irrigate_now: float
    confidence_schedule_soon: float
    confidence_hold_irrigation: float
    top_confidence: float
    low_confidence: bool
    prediction_status: str
    error_flag: bool
    error_message: str
    model_version: str
    confidence: dict[str, float]
    features_used: list[str]


app = FastAPI(
    title="Irrigation Recommendation API",
    description="Predict irrigation actions from soil moisture, soil temperature, air humidity, and zone readings.",
    version="1.1.0",
)


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
    return joblib.load(MODEL_PATH)


def emit_prediction_log(request_row: dict, prediction_payload: dict):
    log_record = build_prediction_log_record(request_row, prediction_payload)
    print("[prediction_log] " + json.dumps(log_record, default=str))


def safe_prediction_response(
    *,
    request_row: dict,
    recommendation: str | None,
    class_confidences: dict[str, float],
    features_used: list[str],
    prediction_status: str,
    error_message: str,
):
    payload = build_prediction_payload(
        recommendation=recommendation,
        class_confidences=class_confidences,
        features_used=features_used,
        prediction_status=prediction_status,
        error_message=error_message,
        model_version=MODEL_VERSION,
    )
    emit_prediction_log(request_row, payload)
    return PredictionResponse(**payload)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH.resolve()),
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    request_row = {
        "moisture": request.moisture,
        "temperature": request.temperature,
        "humidity": request.humidity,
        "zone": request.zone,
    }

    try:
        # Invalid input is tracked as a separate status so adviser reviews can
        # distinguish sensor/request issues from runtime model failures.
        validation_error = validate_prediction_input(
            request.moisture,
            request.temperature,
            request.humidity,
            request.zone,
        )
        if validation_error:
            return safe_prediction_response(
                request_row=request_row,
                recommendation=None,
                class_confidences={},
                features_used=[],
                prediction_status="invalid_input",
                error_message=validation_error,
            )

        feature_columns: list[str] = []

        try:
            bundle = load_bundle()
            model = bundle["model"]
            feature_columns = list(bundle["feature_columns"])

            input_df = pd.DataFrame([request_row]).reindex(columns=feature_columns)
            recommendation = str(model.predict(input_df)[0])

            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Model does not support predict_proba.")

            probabilities = model.predict_proba(input_df)[0]

            if (
                hasattr(model, "named_steps")
                and "classifier" in model.named_steps
                and hasattr(model.named_steps["classifier"], "classes_")
            ):
                labels = model.named_steps["classifier"].classes_
            elif hasattr(model, "classes_"):
                labels = model.classes_
            else:
                raise RuntimeError("Unable to resolve class labels for predict_proba output.")

            class_confidences = build_class_confidence_map(labels, probabilities)

            return safe_prediction_response(
                request_row=request_row,
                recommendation=recommendation,
                class_confidences=class_confidences,
                features_used=feature_columns,
                prediction_status="success",
                error_message="",
            )
        except Exception as exc:
            return safe_prediction_response(
                request_row=request_row,
                recommendation=None,
                class_confidences={},
                features_used=feature_columns,
                prediction_status="model_error",
                error_message=str(exc),
            )
    except Exception as exc:
        return safe_prediction_response(
            request_row=request_row,
            recommendation=None,
            class_confidences={},
            features_used=[],
            prediction_status="api_error",
            error_message=str(exc),
        )
