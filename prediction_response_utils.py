from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

CLASS_LABELS = ("irrigate_now", "schedule_soon", "hold_irrigation")
LOW_CONFIDENCE_THRESHOLD = 0.60


def _coerce_float(name: str, value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, f"{name} is required."

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a number."

    if not math.isfinite(number):
        return None, f"{name} must be a finite number."

    return number, ""


def validate_prediction_input(
    moisture: Any,
    temperature: Any,
    humidity: Any,
    zone: Any,
) -> str:
    moisture_value, moisture_error = _coerce_float("moisture", moisture)
    if moisture_error:
        return moisture_error

    humidity_value, humidity_error = _coerce_float("humidity", humidity)
    if humidity_error:
        return humidity_error

    temperature_value, temperature_error = _coerce_float("temperature", temperature)
    if temperature_error:
        return temperature_error

    if moisture_value is None or not (0 <= moisture_value <= 100):
        return "moisture must be between 0 and 100."

    if humidity_value is None or not (0 <= humidity_value <= 100):
        return "humidity must be between 0 and 100."

    if temperature_value is None or not (-20 <= temperature_value <= 60):
        return "temperature must be between -20 and 60."

    zone_value = "" if zone is None else str(zone).strip()
    if not zone_value:
        return "zone is required and must be non-empty."

    return ""


def build_class_confidence_map(
    labels: Iterable[Any] | None,
    probabilities: Iterable[Any] | None,
) -> dict[str, float]:
    confidence_map = {label: 0.0 for label in CLASS_LABELS}

    if labels is None or probabilities is None:
        return confidence_map

    for label, score in zip(labels, probabilities):
        normalized_label = str(label).strip()
        if normalized_label not in confidence_map:
            continue

        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0

        if not math.isfinite(numeric_score):
            numeric_score = 0.0

        confidence_map[normalized_label] = max(0.0, min(1.0, numeric_score))

    return confidence_map


def build_prediction_payload(
    *,
    recommendation: str | None,
    class_confidences: Mapping[str, Any] | None,
    features_used: list[str],
    prediction_status: str,
    error_message: str,
    model_version: str,
) -> dict[str, Any]:
    normalized_confidences = {
        label: float(class_confidences.get(label, 0.0)) if class_confidences else 0.0
        for label in CLASS_LABELS
    }

    top_confidence = max(normalized_confidences.values()) if normalized_confidences else 0.0

    # Low-confidence responses are flagged because adviser review requires uncertainty
    # to be tracked as an operational "error" signal, even when inference succeeds.
    low_confidence = top_confidence < LOW_CONFIDENCE_THRESHOLD

    final_error_message = (error_message or "").strip()
    if prediction_status == "success":
        if low_confidence and not final_error_message:
            final_error_message = "Low-confidence prediction"
        elif not low_confidence:
            final_error_message = ""

    # Adviser-required "error" includes both hard failures and uncertain predictions.
    error_flag = prediction_status != "success" or low_confidence

    sorted_confidence_items = sorted(
        normalized_confidences.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    compatibility_confidence = {
        label: round(float(score), 4) for label, score in sorted_confidence_items
    }

    return {
        "recommendation": recommendation if prediction_status == "success" else None,
        "confidence_irrigate_now": round(normalized_confidences["irrigate_now"], 4),
        "confidence_schedule_soon": round(normalized_confidences["schedule_soon"], 4),
        "confidence_hold_irrigation": round(normalized_confidences["hold_irrigation"], 4),
        "top_confidence": round(float(top_confidence), 4),
        "low_confidence": low_confidence,
        "prediction_status": prediction_status,
        "error_flag": error_flag,
        "error_message": final_error_message,
        "model_version": model_version,
        "confidence": compatibility_confidence,
        "features_used": list(features_used or []),
    }
