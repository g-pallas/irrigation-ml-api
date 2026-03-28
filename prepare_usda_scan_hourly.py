import argparse
from pathlib import Path

import pandas as pd


COMMON_ALIASES = {
    "date": [
        "Date",
        "date",
    ],
    "time": [
        "Time",
        "time",
    ],
    "moisture_2in": [
        "Soil Moisture Percent -2in",
        "soil_moisture_2in",
        "SMS.I-1:-2 (pct)",
        "SMS -2.0 in",
        "soil moisture percent 2 in",
        "soil moisture 2 in",
    ],
    "temperature_2in": [
        "Soil Temperature -2in",
        "soil_temperature_2in",
        "STO.I-1:-2 (degC)",
        "STO -2.0 in",
        "soil temperature 2 in",
    ],
    "humidity": [
        "Relative Humidity",
        "relative_humidity",
        "RHUMV (%)",
        "RHUM",
        "humidity",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a USDA SCAN hourly export into a model-ready irrigation CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more SCAN hourly CSV or Excel files to combine.",
    )
    parser.add_argument(
        "--output",
        default="prepared_soil_data_scan_hourly.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name or zero-based index if the SCAN file is .xlsx.",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=5,
        help="Number of rows to skip before the actual header row. USDA SCAN exports often need 5.",
    )
    parser.add_argument("--date-col", default=None, help="Override the date column name.")
    parser.add_argument("--time-col", default=None, help="Override the time column name.")
    parser.add_argument(
        "--moisture-col",
        default=None,
        help="Override the 2-inch soil moisture column name.",
    )
    parser.add_argument(
        "--temperature-col",
        default=None,
        help="Override the 2-inch soil temperature column name.",
    )
    parser.add_argument(
        "--humidity-col",
        default=None,
        help="Override the relative humidity column name.",
    )
    parser.add_argument(
        "--zone",
        default="Walnut Gulch #1, AZ",
        help="Zone or station label to place in the output CSV.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "three_class", "duration", "quantile"],
        default="quantile",
        help="How to derive the irrigation target from soil moisture.",
    )
    parser.add_argument(
        "--dry-threshold",
        type=float,
        default=20.0,
        help="Binary/three-class low moisture threshold.",
    )
    parser.add_argument(
        "--wet-threshold",
        type=float,
        default=40.0,
        help="Three-class high moisture threshold.",
    )
    parser.add_argument(
        "--invalid-below",
        type=float,
        default=-50.0,
        help="Drop rows where moisture, temperature, or humidity are below this value. Useful for SCAN sentinels like -99.9.",
    )
    parser.add_argument(
        "--invalid-above",
        type=float,
        default=None,
        help="Optional upper bound for invalid values across numeric sensor columns.",
    )
    return parser.parse_args()


def clean_column_name(name: str):
    return " ".join(name.strip().split())


def normalize(name: str):
    return clean_column_name(name).lower().replace("-", "_")


def load_table(path: Path, sheet_name: str | None, skiprows: int):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, skiprows=skiprows)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        selected_sheet = sheet_name
        if selected_sheet is not None:
            try:
                selected_sheet = int(selected_sheet)
            except ValueError:
                pass
        df = pd.read_excel(path, sheet_name=selected_sheet, skiprows=skiprows)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .xls.")

    df.columns = [clean_column_name(str(column)) for column in df.columns]
    return df


def find_column(df: pd.DataFrame, requested_name: str | None, alias_key: str):
    if requested_name:
        cleaned_requested_name = clean_column_name(requested_name)
        if cleaned_requested_name not in df.columns:
            raise ValueError(
                f"Column '{requested_name}' not found. Available columns: {list(df.columns)}"
            )
        return cleaned_requested_name

    normalized_lookup = {normalize(column): column for column in df.columns}
    for alias in COMMON_ALIASES[alias_key]:
        match = normalized_lookup.get(normalize(alias))
        if match:
            return match
    return None


def apply_invalid_filters(prepared: pd.DataFrame, invalid_below: float | None, invalid_above: float | None):
    numeric_columns = ["moisture", "temperature", "humidity"]
    filtered = prepared.copy()

    if invalid_below is not None:
        for column in numeric_columns:
            filtered = filtered[filtered[column] >= invalid_below]

    if invalid_above is not None:
        for column in numeric_columns:
            filtered = filtered[filtered[column] <= invalid_above]

    filtered = filtered[
        filtered["moisture"].between(0, 100)
        & filtered["humidity"].between(0, 100)
        & filtered["temperature"].between(-40, 80)
    ]
    return filtered


def derive_recommendation(
    moisture: float,
    temperature: float | None,
    humidity: float | None,
    label_mode: str,
    dry_threshold: float,
    wet_threshold: float,
):
    if pd.isna(moisture):
        return None

    if label_mode == "duration":
        humidity_value = 55 if pd.isna(humidity) else humidity
        if moisture < dry_threshold:
            return 15 if humidity_value < 45 else 12
        if moisture < 30:
            return 10 if humidity_value < 45 else 8
        if moisture < wet_threshold:
            return 5
        return 0

    humidity_adjustment = 0.0
    if not pd.isna(humidity):
        if humidity >= 85:
            humidity_adjustment += 2.0
        elif humidity >= 70:
            humidity_adjustment += 1.0
        elif humidity <= 20:
            humidity_adjustment -= 2.0
        elif humidity <= 30:
            humidity_adjustment -= 1.0

    temperature_adjustment = 0.0
    if not pd.isna(temperature):
        if temperature >= 32:
            temperature_adjustment -= 2.0
        elif temperature >= 26:
            temperature_adjustment -= 1.0
        elif temperature <= 8:
            temperature_adjustment += 1.0

    effective_dry_threshold = dry_threshold + humidity_adjustment + temperature_adjustment
    effective_wet_threshold = wet_threshold + humidity_adjustment + temperature_adjustment

    if label_mode == "binary":
        return "irrigate_now" if moisture < effective_dry_threshold else "hold_irrigation"

    if moisture < effective_dry_threshold:
        return "irrigate_now"
    if moisture < effective_wet_threshold:
        return "schedule_soon"
    return "hold_irrigation"


def derive_quantile_recommendations(prepared: pd.DataFrame):
    if prepared.empty:
        return pd.Series(dtype="object")

    low_cutoff = prepared["moisture"].quantile(1 / 3)
    high_cutoff = prepared["moisture"].quantile(2 / 3)

    def classify(row):
        if row["moisture"] < low_cutoff:
            return "irrigate_now"
        if row["moisture"] < high_cutoff:
            return "schedule_soon"
        return "hold_irrigation"

    return prepared.apply(classify, axis=1)


def build_timestamp(date_series: pd.Series, time_series: pd.Series | None):
    date_text = date_series.astype(str).str.strip()
    if time_series is None:
        return pd.to_datetime(date_text, errors="coerce")
    time_text = time_series.astype(str).str.strip()
    combined = date_text + " " + time_text
    return pd.to_datetime(combined, errors="coerce")


def main():
    args = parse_args()

    frames = []
    resolved_inputs = []
    for input_arg in args.input:
        input_path = Path(input_arg)
        if not input_path.exists():
            raise FileNotFoundError(f"SCAN dataset file not found: {input_path}")
        frames.append(load_table(input_path, args.sheet, args.skiprows))
        resolved_inputs.append(input_path.resolve())

    df = pd.concat(frames, ignore_index=True)

    date_col = find_column(df, args.date_col, "date")
    time_col = find_column(df, args.time_col, "time")
    moisture_col = find_column(df, args.moisture_col, "moisture_2in")
    temperature_col = find_column(df, args.temperature_col, "temperature_2in")
    humidity_col = find_column(df, args.humidity_col, "humidity")

    missing = [
        name
        for name, value in {
            "date": date_col,
            "time": time_col,
            "moisture_2in": moisture_col,
            "temperature_2in": temperature_col,
            "humidity": humidity_col,
        }.items()
        if value is None and name != "time"
    ]
    if missing:
        raise ValueError(
            f"Could not detect required SCAN columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    timestamp = build_timestamp(df[date_col], df[time_col] if time_col else None)

    prepared = pd.DataFrame(
        {
            "timestamp": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "zone": args.zone,
            "moisture": pd.to_numeric(df[moisture_col], errors="coerce"),
            "temperature": pd.to_numeric(df[temperature_col], errors="coerce"),
            "humidity": pd.to_numeric(df[humidity_col], errors="coerce"),
        }
    )
    prepared = prepared.dropna(subset=["timestamp", "moisture", "temperature", "humidity"])
    rows_before_filter = len(prepared)
    prepared = apply_invalid_filters(prepared, args.invalid_below, args.invalid_above)
    rows_removed = rows_before_filter - len(prepared)

    if args.label_mode == "quantile":
        prepared["recommendation"] = derive_quantile_recommendations(prepared)
    else:
        prepared["recommendation"] = prepared.apply(
            lambda row: derive_recommendation(
                moisture=row["moisture"],
                temperature=row["temperature"],
                humidity=row["humidity"],
                label_mode=args.label_mode,
                dry_threshold=args.dry_threshold,
                wet_threshold=args.wet_threshold,
            ),
            axis=1,
        )

    output_columns = [
        "timestamp",
        "zone",
        "moisture",
        "temperature",
        "humidity",
        "recommendation",
    ]
    if args.label_mode == "duration":
        output_columns[-1] = "irrigation_duration"
        prepared = prepared.rename(columns={"recommendation": "irrigation_duration"})

    prepared = prepared[output_columns].sort_values("timestamp").reset_index(drop=True)

    output_path = Path(args.output)
    prepared.to_csv(output_path, index=False)

    print("USDA SCAN hourly preparation complete.")
    print(f"Rows prepared: {len(prepared)}")
    print(f"Rows removed by invalid-value filtering: {rows_removed}")
    print(f"Output: {output_path.resolve()}")
    print("Input files:")
    for path in resolved_inputs:
        print(f"- {path}")
    print(
        f"Detected columns: date={date_col}, time={time_col}, moisture={moisture_col}, "
        f"temperature={temperature_col}, humidity={humidity_col}"
    )
    target_col = "irrigation_duration" if args.label_mode == "duration" else "recommendation"
    print(f"{target_col} counts:")
    print(prepared[target_col].value_counts(dropna=False).to_string())
    print("\nPreview:")
    print(prepared.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
