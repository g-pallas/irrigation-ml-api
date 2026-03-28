import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the irrigation recommendation model and export appendix-ready artifacts."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the labeled CSV used for evaluation.",
    )
    parser.add_argument(
        "--model",
        default="irrigation_model_scan_hourly.pkl",
        help="Path to the trained model bundle.",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation_outputs_scan_hourly",
        help="Directory where reports and images will be saved.",
    )
    return parser.parse_args()


def save_confusion_matrix_image(cm, labels, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Irrigation Recommendation Confusion Matrix")

    max_value = cm.max() if cm.size else 0
    threshold = max_value / 2 if max_value else 0
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            value = cm[row_index, col_index]
            color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target_column = bundle["target_column"]
    labels = bundle["labels"]

    missing = [column for column in feature_columns + [target_column] if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in evaluation CSV: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    X = df[feature_columns]
    y_true = df[target_column].astype(str)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    class_counts = y_true.value_counts().reindex(labels, fill_value=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    counts_csv_path = out_dir / "class_distribution.csv"
    counts_df = class_counts.rename_axis("recommendation").reset_index(name="count")
    counts_df.to_csv(counts_csv_path, index=False)

    cm_image_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix_image(cm, labels, cm_image_path)

    report_path = out_dir / "metrics_report.txt"
    report_lines = [
        "Irrigation Recommendation Model Evaluation",
        "",
        f"Dataset: {csv_path.resolve()}",
        f"Model: {model_path.resolve()}",
        f"Rows evaluated: {len(df)}",
        f"Features used: {feature_columns}",
        f"Accuracy: {accuracy:.4f}",
        "",
        "Class Distribution",
        counts_df.to_string(index=False),
        "",
        "Classification Report",
        report_text,
        "Appendix Note",
        (
            "These metrics were computed on a labeled dataset prepared for the capstone "
            "pipeline. If recommendation labels were derived from threshold or quantile rules, "
            "the scores mainly validate pipeline consistency rather than independent field accuracy."
        ),
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Evaluation complete.")
    print(f"Rows evaluated: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved class distribution to: {counts_csv_path.resolve()}")
    print(f"Saved confusion matrix image to: {cm_image_path.resolve()}")
    print(f"Saved metrics report to: {report_path.resolve()}")
    print("\nClass distribution:")
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
