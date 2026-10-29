import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIFT_RESULTS_PATH = PROJECT_ROOT / "artifacts" / "drift_results.json"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "fraud_model.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "evaluation_metrics.json"
TRAINING_SCRIPT = PROJECT_ROOT / "ml" / "train_model.py"
STATUS_PATH = PROJECT_ROOT / "artifacts" / "retraining_status.json"

DRIFT_THRESHOLD = 0.30
MINIMUM_ROC_AUC = 0.90
MINIMUM_PR_AUC = 0.10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = STATUS_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(status, indent=4),
        encoding="utf-8",
    )
    temporary_path.replace(STATUS_PATH)


def validate_drift_results(results):
    required_keys = {
        "drift_share",
        "drift_detected",
        "retraining_recommended",
    }

    missing = required_keys - results.keys()

    if missing:
        raise ValueError(
            f"Drift results are missing fields: {sorted(missing)}"
        )

    drift_share = results["drift_share"]

    if drift_share is None:
        raise ValueError(
            "Drift share is unavailable. Run the drift monitor successfully first."
        )

    drift_share = float(drift_share)

    if not 0.0 <= drift_share <= 1.0:
        raise ValueError("Drift share must be between 0 and 1.")

    return drift_share


def train_candidate_model():
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    logger.info("Starting model training and MLflow tracking.")

    result = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )

    if result.stdout:
        logger.info("Training output:\n%s", result.stdout[-12000:])

    if result.stderr:
        logger.info("Training messages:\n%s", result.stderr[-6000:])

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed with exit code {result.returncode}."
        )

    logger.info("Training script completed successfully.")


def validate_model_metrics():
    metrics = read_json(METRICS_PATH)

    roc_auc = float(metrics.get("roc_auc", 0.0))
    pr_auc = float(metrics.get("pr_auc_average_precision", 0.0))

    if not 0.0 <= roc_auc <= 1.0:
        raise ValueError(f"Invalid ROC-AUC: {roc_auc}")

    if not 0.0 <= pr_auc <= 1.0:
        raise ValueError(f"Invalid PR-AUC: {pr_auc}")

    if roc_auc < MINIMUM_ROC_AUC:
        raise ValueError(
            f"Candidate ROC-AUC {roc_auc:.4f} is below "
            f"the minimum {MINIMUM_ROC_AUC:.4f}."
        )

    if pr_auc < MINIMUM_PR_AUC:
        raise ValueError(
            f"Candidate PR-AUC {pr_auc:.4f} is below "
            f"the minimum {MINIMUM_PR_AUC:.4f}."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Training finished, but the model artifact is missing."
        )

    return {
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
    }


def main():
    logger.info("Checking whether model retraining is recommended.")

    timestamp = datetime.now(timezone.utc).isoformat()

    status = {
        "checked_at": timestamp,
        "retraining_attempted": False,
        "retraining_completed": False,
        "reason": None,
    }

    try:
        drift_results = read_json(DRIFT_RESULTS_PATH)
        drift_share = validate_drift_results(drift_results)

        logger.info("Measured feature drift share: %.2f%%", drift_share * 100)

        should_retrain = (
            drift_share >= DRIFT_THRESHOLD
            or drift_results["retraining_recommended"] is True
        )

        if not should_retrain:
            status["reason"] = "Drift is below the retraining threshold."
            write_status(status)
            logger.info(status["reason"])
            return

        if not drift_results["drift_detected"]:
            logger.warning(
                "Retraining was recommended, but drift_detected is false. "
                "Proceeding because the configured share threshold was reached."
            )

        status["retraining_attempted"] = True
        status["reason"] = "Drift threshold reached."
        write_status(status)

        train_candidate_model()

        candidate_metrics = validate_model_metrics()

        status["retraining_completed"] = True
        status["candidate_metrics"] = candidate_metrics
        status["reason"] = (
            "Training completed and model metrics passed configured checks."
        )
        write_status(status)

        logger.info(status["reason"])

    except Exception as exc:
        status["reason"] = str(exc)
        status["error"] = type(exc).__name__
        write_status(status)

        logger.exception("Retraining workflow failed.")
        raise


if __name__ == "__main__":
    main()