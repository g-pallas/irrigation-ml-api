import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Run irrigation recommendation prediction from sensor values."
    )
    parser.add_argument("--model", default="irrigation_model_scan_hourly.pkl")
    parser.add_argument("--moisture", required=True, type=float)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--humidity", type=float, default=None)
    parser.add_argument("--soil-ph", dest="soil_ph", type=float, default=None)
    parser.add_argument("--zone", default=None)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    row = {
        "moisture": args.moisture,
        "temperature": args.temperature,
        "humidity": args.humidity,
        "soil_ph": args.soil_ph,
        "zone": args.zone,
    }

    input_df = pd.DataFrame([row]).reindex(columns=feature_columns)
    prediction = model.predict(input_df)[0]

    print("Input:", row)
    print("Recommendation:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        labels = model.named_steps["classifier"].classes_
        ranked = sorted(
            zip(labels, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        print("Confidence:")
        for label, score in ranked:
            print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
