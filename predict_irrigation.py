import argparse
import json
import os

import joblib
import pandas as pd

from prediction_log_payload import build_prediction_log_record
from prediction_response_utils import (
    build_class_confidence_map,
    build_prediction_payload,
    validate_prediction_input,
)

MODEL_VERSION = os.getenv("IRRIGATION_MODEL_VERSION", "scan_hourly_v1")


def run_prediction(args) -> dict:
    request_row = {
        "moisture": args.moisture,
        "temperature": args.temperature,
        "humidity": args.humidity,
        "zone": args.zone,
    }

    validation_error = validate_prediction_input(
        args.moisture,
        args.temperature,
        args.humidity,
        args.zone,
    )
    if validation_error:
        return build_prediction_payload(
            recommendation=None,
            class_confidences={},
            features_used=[],
            prediction_status="invalid_input",
            error_message=validation_error,
            model_version=MODEL_VERSION,
        )

    feature_columns: list[str] = []

    try:
        bundle = joblib.load(args.model)
        model = bundle["model"]
        feature_columns = list(bundle["feature_columns"])

        row = {
            "moisture": args.moisture,
            "temperature": args.temperature,
            "humidity": args.humidity,
            "soil_ph": args.soil_ph,
            "zone": args.zone,
        }

        input_df = pd.DataFrame([row]).reindex(columns=feature_columns)
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

        return build_prediction_payload(
            recommendation=recommendation,
            class_confidences=class_confidences,
            features_used=feature_columns,
            prediction_status="success",
            error_message="",
            model_version=MODEL_VERSION,
        )
    except Exception as exc:
        return build_prediction_payload(
            recommendation=None,
            class_confidences={},
            features_used=feature_columns,
            prediction_status="model_error",
            error_message=str(exc),
            model_version=MODEL_VERSION,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run irrigation recommendation prediction from sensor values."
    )
    parser.add_argument("--model", default="irrigation_model_scan_hourly.pkl")
    parser.add_argument("--moisture", required=True, type=float)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--humidity", required=True, type=float)
    parser.add_argument("--soil-ph", dest="soil_ph", type=float, default=None)
    parser.add_argument("--zone", required=True)
    args = parser.parse_args()

    request_row = {
        "moisture": args.moisture,
        "temperature": args.temperature,
        "humidity": args.humidity,
        "zone": args.zone,
    }

    prediction = run_prediction(args)

    print("Input:", request_row)
    print("Recommendation:", prediction.get("recommendation"))
    print("Prediction status:", prediction.get("prediction_status"))
    print("Model version:", prediction.get("model_version"))
    print("Top confidence:", prediction.get("top_confidence"))
    print("Low confidence:", prediction.get("low_confidence"))
    print("Error flag:", prediction.get("error_flag"))

    error_message = str(prediction.get("error_message", "") or "")
    if error_message:
        print("Error message:", error_message)

    print("Class confidences:")
    print(f"  irrigate_now: {prediction.get('confidence_irrigate_now', 0.0):.4f}")
    print(f"  schedule_soon: {prediction.get('confidence_schedule_soon', 0.0):.4f}")
    print(f"  hold_irrigation: {prediction.get('confidence_hold_irrigation', 0.0):.4f}")

    print("Features used:", prediction.get("features_used", []))

    log_record = build_prediction_log_record(request_row, prediction)
    print("Log payload:")
    print(json.dumps(log_record, indent=2))


if __name__ == "__main__":
    main()
