
import argparse
import json
import logging
import uuid
from pathlib import Path

import joblib
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
DATASET_PATH = ROOT / "datasets" / "creditcard.csv"

TOPIC = "fraud-transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
SEND_TIMEOUT_SECONDS = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_sample(sample_type):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        raise ValueError(
            "The saved model does not expose feature_names_in_. "
            "Check the model's expected input features."
        )

    feature_names = list(feature_names)

    if not feature_names:
        raise ValueError("The model has no recorded input features.")

    df = pd.read_csv(DATASET_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    missing_columns = [
        name for name in feature_names if name not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing model features: {missing_columns}"
        )

    target_class = 1 if sample_type == "fraud" else 0
    samples = df.loc[df["Class"] == target_class]

    if samples.empty:
        raise ValueError(
            f"No transactions found for sample type '{sample_type}'."
        )

    row = samples.iloc[0]
    features = {}

    for name in feature_names:
        value = pd.to_numeric(row[name], errors="raise")

        if pd.isna(value):
            raise ValueError(f"Feature '{name}' contains a missing value.")

        features[name] = float(value)

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        **features,
    }

    return transaction, target_class, len(feature_names)


def publish_transaction(sample_type, bootstrap_servers=BOOTSTRAP_SERVERS):
    transaction, actual_class, feature_count = load_sample(sample_type)

    event = {
        "event_id": str(uuid.uuid4()),
        "transaction": transaction,
        "sample_type": sample_type,
    }

    producer = None

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda value: json.dumps(
                value,
                allow_nan=False,
            ).encode("utf-8"),
            acks="all",
            retries=5,
            request_timeout_ms=10000,
            max_block_ms=15000,
        )

        metadata = producer.send(TOPIC, event).get(
            timeout=SEND_TIMEOUT_SECONDS
        )

        logger.info("Transaction published successfully.")
        logger.info("Event ID: %s", event["event_id"])
        logger.info("Topic: %s", metadata.topic)
        logger.info("Partition: %s", metadata.partition)
        logger.info("Offset: %s", metadata.offset)
        logger.info("Sample type: %s", sample_type)
        logger.info("Dataset label: %s", actual_class)
        logger.info("Features sent: %s", feature_count)

        return event

    except (KafkaError, OSError, ValueError, TimeoutError) as exc:
        logger.error("Failed to publish transaction: %s", exc)
        raise

    finally:
        if producer is not None:
            try:
                producer.flush(timeout=10)
            finally:
                producer.close(timeout=10)


def main():
    parser = argparse.ArgumentParser(
        description="Publish a sample transaction to Kafka."
    )

    parser.add_argument(
        "--type",
        choices=["legitimate", "fraud"],
        default="legitimate",
        help="Select a labeled transaction from the dataset.",
    )

    parser.add_argument(
        "--bootstrap-servers",
        default=BOOTSTRAP_SERVERS,
        help="Comma-separated Kafka broker addresses.",
    )

    args = parser.parse_args()

    try:
        publish_transaction(
            sample_type=args.type,
            bootstrap_servers=args.bootstrap_servers,
        )
    except (FileNotFoundError, ValueError, KafkaError, OSError, TimeoutError) as exc:
        logger.error("Transaction publishing failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

