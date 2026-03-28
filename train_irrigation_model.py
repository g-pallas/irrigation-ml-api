import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "recommendation"
DEFAULT_NUMERIC_FEATURES = ["moisture", "temperature", "humidity", "soil_ph"]
DEFAULT_CATEGORICAL_FEATURES = ["zone"]


def validate_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def detect_features(df):
    numeric_features = [
        column
        for column in DEFAULT_NUMERIC_FEATURES
        if column in df.columns and df[column].notna().any()
    ]
    categorical_features = [
        column
        for column in DEFAULT_CATEGORICAL_FEATURES
        if column in df.columns and df[column].notna().any()
    ]
    if not numeric_features:
        raise ValueError(
            "Dataset must include at least one numeric feature such as "
            "'moisture' or 'temperature'."
        )
    return numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )


def can_stratify(labels):
    return labels.value_counts().min() >= 2


def main():
    parser = argparse.ArgumentParser(
        description="Train an irrigation recommendation classifier."
    )
    parser.add_argument(
        "--csv",
        default="prepared_soil_data_scan_hourly.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--model-out",
        default="irrigation_model_scan_hourly.pkl",
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data to use for testing. Default: 0.25",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, ["moisture", "temperature", TARGET_COLUMN])

    if len(df) < 12:
        raise ValueError(
            "Dataset is too small. Add at least 12 rows before training."
        )

    numeric_features, categorical_features = detect_features(df)
    feature_columns = numeric_features + categorical_features
    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(str)

    split_kwargs = {
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    if can_stratify(y):
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining complete.")
    print(f"Rows used: {len(df)}")
    print(f"Features used: {feature_columns}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    model_out = Path(args.model_out)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "labels": sorted(y.unique().tolist()),
        },
        model_out,
    )
    print(f"Saved trained model to: {model_out.resolve()}")

    sample_input = pd.DataFrame(
        [
            {
                "moisture": 24,
                "temperature": 33,
                "humidity": 62,
                "soil_ph": 6.4,
                "zone": "north_plot",
            }
        ]
    )
    sample_input = sample_input.reindex(columns=feature_columns)
    sample_prediction = model.predict(sample_input)[0]
    print("\nSample prediction")
    print(sample_input.to_dict(orient="records")[0])
    print(f"Recommended action: {sample_prediction}")


if __name__ == "__main__":
    main()
