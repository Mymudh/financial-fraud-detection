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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_sample(sample_type):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = list(model.feature_names_in_)

    df = pd.read_csv(DATASET_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if sample_type == "fraud":
        samples = df[df["Class"] == 1]
    else:
        samples = df[df["Class"] == 0]

    if samples.empty:
        raise ValueError(f"No {sample_type} samples found in the dataset.")

    row = samples.iloc[0]
    transaction = {}

    for name in feature_names:
        if name not in row.index:
            raise ValueError(
                f"Model feature '{name}' is missing from the dataset."
            )

        transaction[name] = float(row[name])

    transaction["transaction_id"] = str(uuid.uuid4())

    return model, feature_names, transaction, int(row["Class"])


def main():
    parser = argparse.ArgumentParser(
        description="Publish a sample transaction to Kafka."
    )

    parser.add_argument(
        "--type",
        choices=["legitimate", "fraud"],
        default="legitimate",
        help="Select a labeled dataset example to publish.",
    )

    args = parser.parse_args()

    sample_type = args.type

    try:
        model, feature_names, transaction, actual_class = load_sample(
            sample_type
        )

        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            acks="all",
            retries=5,
            request_timeout_ms=10000,
        )

        event = {
            "event_id": str(uuid.uuid4()),
            "transaction": transaction,
            "sample_type": sample_type,
        }

        try:
            metadata = producer.send(TOPIC, event).get(timeout=15)
            logger.info("Transaction published successfully.")
            logger.info("Topic: %s", metadata.topic)
            logger.info("Partition: %s", metadata.partition)
            logger.info("Offset: %s", metadata.offset)
            logger.info("Sample type: %s", sample_type)
            logger.info("Actual dataset label: %s", actual_class)
            logger.info("Features sent: %s", len(feature_names))
        finally:
            producer.flush(timeout=10)
            producer.close(timeout=10)

    except (FileNotFoundError, ValueError, KafkaError, OSError) as exc:
        logger.error("Failed to publish transaction: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()