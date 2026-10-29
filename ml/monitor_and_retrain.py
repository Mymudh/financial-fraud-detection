import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "evidently_debug.json"
STATUS_PATH = ARTIFACTS_DIR / "drift_status.json"
RETRAIN_SCRIPT = ROOT / "ml" / "auto_retrain.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def load_drift_report():
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Drift report not found: {REPORT_PATH}. "
            "Run python ml/inspect_drift.py first."
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("Invalid report: expected a JSON object.")

    metrics = report.get("metrics")

    if not isinstance(metrics, list):
        raise ValueError(
            "The report does not contain a metrics list. "
            "Check ml/inspect_drift.py."
        )

    logger.info("Drift report loaded successfully.")
    logger.info("Metrics found: %s", len(metrics))

    return report


def evaluate_drift(report):
    results = []

    for metric in report.get("metrics", []):
        if not isinstance(metric, dict):
            continue

        config = metric.get("config") or {}
        metric_name = metric.get("metric_name", "")
        column = config.get("column")
        threshold = config.get("threshold")
        value = metric.get("value")

        if column is None or threshold is None or value is None:
            continue

        try:
            threshold = float(threshold)
            value = float(value)
        except (TypeError, ValueError):
            continue

        if not all(map(lambda x: x == x and abs(x) != float("inf"), [threshold, value])):
            continue

        drift_detected = value > threshold

        results.append({
            "column": str(column),
            "metric_name": str(metric_name),
            "value": value,
            "threshold": threshold,
            "drift_detected": drift_detected
        })

    if not results:
        raise ValueError(
            "No valid column drift metrics were found in the report. "
            "Check the metric structure produced by inspect_drift.py."
        )

    return results


def run_retraining():
    if not RETRAIN_SCRIPT.is_file():
        return {
            "attempted": False,
            "success": False,
            "message": f"Retraining script not found: {RETRAIN_SCRIPT}"
        }

    logger.warning("Drift detected. Starting model retraining.")

    try:
        completed = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False
        )

        output = (completed.stdout or "")[-5000:]
        errors = (completed.stderr or "")[-5000:]

        if completed.returncode == 0:
            logger.info("Retraining script completed successfully.")
            return {
                "attempted": True,
                "success": True,
                "return_code": completed.returncode,
                "stdout": output,
                "stderr": errors,
                "message": "Retraining script completed successfully."
            }

        logger.error(
            "Retraining failed with exit code %s.",
            completed.returncode
        )

        return {
            "attempted": True,
            "success": False,
            "return_code": completed.returncode,
            "stdout": output,
            "stderr": errors,
            "message": "Retraining script returned an error."
        }

    except subprocess.TimeoutExpired:
        logger.exception("Retraining timed out.")
        return {
            "attempted": True,
            "success": False,
            "message": "Retraining exceeded the 30-minute timeout."
        }

    except OSError as error:
        logger.exception("Unable to start retraining.")
        return {
            "attempted": True,
            "success": False,
            "message": str(error)
        }


def save_status(results, retraining=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    drifted_columns = [
        result["column"]
        for result in results
        if result["drift_detected"]
    ]

    status = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(REPORT_PATH),
        "metrics_checked": len(results),
        "drift_detected": bool(drifted_columns),
        "drifted_columns": drifted_columns,
        "retraining_recommended": bool(drifted_columns),
        "retraining": retraining or {
            "attempted": False,
            "success": False,
            "message": "Retraining was not triggered."
        },
        "results": results
    }

    temporary_path = STATUS_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2)

    temporary_path.replace(STATUS_PATH)

    logger.info("Monitoring status saved to: %s", STATUS_PATH)

    return status


def main():
    try:
        report = load_drift_report()
        results = evaluate_drift(report)

        drifted_columns = [
            result["column"]
            for result in results
            if result["drift_detected"]
        ]

        retraining = {
            "attempted": False,
            "success": False,
            "message": "No drift detected; retraining was not triggered."
        }

        if drifted_columns:
            retraining = run_retraining()

        status = save_status(results, retraining)

        print("\n========== DRIFT MONITORING RESULT ==========")
        print(json.dumps(status, indent=2))

        if status["drift_detected"]:
            logger.warning(
                "Drift detected in: %s",
                ", ".join(status["drifted_columns"])
            )
            if not status["retraining"]["success"]:
                logger.error(
                    "Retraining did not complete successfully: %s",
                    status["retraining"]["message"]
                )
        else:
            logger.info(
                "No configured drift thresholds were exceeded."
            )

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()