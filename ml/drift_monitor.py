import json
import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "creditcard.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "drift_report.html"
RESULT_PATH = ARTIFACTS_DIR / "drift_results.json"

TARGET_COLUMN = "Class"
REFERENCE_SIZE = 10000
CURRENT_SIZE = 5000
DRIFT_THRESHOLD = 0.30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_transaction_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    data = data.dropna().copy()

    feature_columns = [
        column
        for column in data.columns
        if column != TARGET_COLUMN
    ]

    required_rows = REFERENCE_SIZE + CURRENT_SIZE

    if len(data) < required_rows:
        raise ValueError(
            f"Dataset requires at least {required_rows} rows; "
            f"found {len(data)}."
        )

    return data, feature_columns


def extract_drift_metrics(report_dict, feature_columns):
    metrics = report_dict.get("metrics", [])

    if not isinstance(metrics, list):
        raise ValueError(
            "Unexpected Evidently report structure: "
            "'metrics' is not a list."
        )

    drift_results = {}

    for metric in metrics:
        if not isinstance(metric, dict):
            continue

        config = metric.get("config", {})

        if not isinstance(config, dict):
            continue

        metric_type = config.get("type", "")
        column = config.get("column")
        metric_name = config.get("metric_name", "")

        if (
            "ValueDrift" not in metric_type
            and "ValueDrift(" not in metric_name
        ):
            continue

        if column not in feature_columns:
            continue

        value = metric.get("value")

        if isinstance(value, dict):
            value = value.get("value")

        try:
            drift_score = float(value)
        except (TypeError, ValueError):
            continue

        threshold = config.get("threshold", 0.1)

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.1

        drift_results[column] = {
            "drift_score": drift_score,
            "threshold": threshold,
            "drift_detected": drift_score >= threshold,
            "method": config.get("method"),
        }

    missing_features = [
        column
        for column in feature_columns
        if column not in drift_results
    ]

    if missing_features:
        logger.warning(
            "Evidently did not provide extractable ValueDrift "
            "metrics for %s features: %s",
            len(missing_features),
            missing_features[:10],
        )

    if not drift_results:
        raise ValueError(
            "No per-feature ValueDrift metrics were extracted. "
            "Inspect artifacts/evidently_debug.json and update "
            "the extraction logic to match its structure."
        )

    drifted_count = sum(
        result["drift_detected"]
        for result in drift_results.values()
    )

    drift_share = drifted_count / len(drift_results)

    return drift_results, drift_share


def generate_drift_report():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading transaction dataset.")

    data, feature_columns = load_transaction_data()

    reference_data = data.iloc[:REFERENCE_SIZE][
        feature_columns
    ].copy()

    current_data = data.iloc[
        REFERENCE_SIZE:REFERENCE_SIZE + CURRENT_SIZE
    ][feature_columns].copy()

    logger.info(
        "Reference rows: %s; current rows: %s; features: %s",
        len(reference_data),
        len(current_data),
        len(feature_columns),
    )

    report = Report(
        metrics=[DataDriftPreset()]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    result.save_html(str(REPORT_PATH))

    report_dict = result.dict()

    drift_results, drift_share = extract_drift_metrics(
        report_dict,
        feature_columns,
    )

    drift_detected = drift_share >= DRIFT_THRESHOLD

    results = {
        "reference_rows": len(reference_data),
        "current_rows": len(current_data),
        "feature_count": len(feature_columns),
        "features_with_metrics": len(drift_results),
        "drifted_feature_count": sum(
            item["drift_detected"]
            for item in drift_results.values()
        ),
        "drift_share": drift_share,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
        "retraining_recommended": drift_detected,
        "features": drift_results,
        "html_report": str(REPORT_PATH),
    }

    RESULT_PATH.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    logger.info("Drift monitoring completed.")
    logger.info("Drift share: %.4f", drift_share)
    logger.info("Drift detected: %s", drift_detected)
    logger.info("HTML report: %s", REPORT_PATH)
    logger.info("JSON results: %s", RESULT_PATH)

    print(json.dumps(results, indent=4))

    return results


if __name__ == "__main__":
    generate_drift_report()