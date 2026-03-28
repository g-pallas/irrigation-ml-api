import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "zone",
    "moisture",
    "temperature",
    "humidity",
    "irrigation_amount",
    "recommendation",
]
STATIONS = ["SA01", "SAP01"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the Mendeley DailyAverageSensedData workbook into a model-ready CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to DailyAverageSensedData1.xlsx",
    )
    parser.add_argument(
        "--sheet",
        default="SensedData",
        help="Sheet name to load from the workbook.",
    )
    parser.add_argument(
        "--output",
        default="prepared_soil_data_mendeley.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--soil-scale",
        type=float,
        default=0.01,
        help="Scale factor applied to SOIL values. Default 0.01 converts values like 6160 to 61.60.",
    )
    parser.add_argument(
        "--water-threshold",
        type=float,
        default=0.0,
        help="Threshold above which irrigation is considered applied.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "three_class"],
        default="binary",
        help="How to generate the recommendation target from irrigation amount.",
    )
    return parser.parse_args()


def infer_recommendation(irrigation_amount, label_mode):
    if pd.isna(irrigation_amount):
        return None

    if label_mode == "binary":
        return "irrigate_now" if irrigation_amount > 0 else "hold_irrigation"

    if irrigation_amount <= 0:
        return "hold_irrigation"
    if irrigation_amount < 0.5:
        return "schedule_soon"
    return "irrigate_now"


def build_station_frame(df, station, soil_scale, label_mode):
    required_columns = [
        "Date",
        f"{station}-SOIL",
        f"{station}-STC",
        f"{station}-HUM",
        f"Water{station}",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for station {station}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    station_df = pd.DataFrame(
        {
            "date": df["Date"],
            "zone": station,
            "moisture": pd.to_numeric(df[f"{station}-SOIL"], errors="coerce") * soil_scale,
            "temperature": pd.to_numeric(df[f"{station}-STC"], errors="coerce"),
            "humidity": pd.to_numeric(df[f"{station}-HUM"], errors="coerce"),
            "irrigation_amount": pd.to_numeric(df[f"Water{station}"], errors="coerce"),
        }
    )
    station_df["recommendation"] = station_df["irrigation_amount"].apply(
        lambda value: infer_recommendation(value, label_mode)
    )
    return station_df


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    df = pd.read_excel(input_path, sheet_name=args.sheet)

    frames = [
        build_station_frame(df, station, args.soil_scale, args.label_mode)
        for station in STATIONS
    ]
    prepared = pd.concat(frames, ignore_index=True)
    prepared = prepared.dropna(
        subset=["date", "moisture", "temperature", "humidity", "irrigation_amount"]
    )
    prepared = prepared[OUTPUT_COLUMNS]

    output_path = Path(args.output)
    prepared.to_csv(output_path, index=False)

    print("Mendeley daily-average preparation complete.")
    print(f"Rows prepared: {len(prepared)}")
    print(f"Output: {output_path.resolve()}")
    print("Recommendation counts:")
    print(prepared["recommendation"].value_counts(dropna=False).to_string())
    print("\nPreview:")
    print(prepared.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
