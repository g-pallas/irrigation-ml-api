from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_prediction_log_record(
    request_data: Mapping[str, Any],
    prediction_response: Mapping[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    event_time = timestamp or datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": event_time,
        "zone": str(request_data.get("zone", "") or ""),
        "moisture": _to_float(request_data.get("moisture")),
        "temperature": _to_float(request_data.get("temperature")),
        "humidity": _to_float(request_data.get("humidity")),
        "recommendation": prediction_response.get("recommendation"),
        "confidence_irrigate_now": _to_float(prediction_response.get("confidence_irrigate_now")),
        "confidence_schedule_soon": _to_float(prediction_response.get("confidence_schedule_soon")),
        "confidence_hold_irrigation": _to_float(prediction_response.get("confidence_hold_irrigation")),
        "top_confidence": _to_float(prediction_response.get("top_confidence")),
        "low_confidence": bool(prediction_response.get("low_confidence", False)),
        "prediction_status": str(prediction_response.get("prediction_status", "api_error") or "api_error"),
        "error_flag": bool(prediction_response.get("error_flag", True)),
        "error_message": str(prediction_response.get("error_message", "") or ""),
        "model_version": str(prediction_response.get("model_version", "") or ""),
    }
