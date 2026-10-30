import argparse
import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "fraud-transactions"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

_producer = None


def get_producer():
    global _producer

    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            value_serializer=lambda value: json.dumps(
                value,
                allow_nan=False
            ).encode("utf-8"),
            acks="all",
            retries=5,
            request_timeout_ms=15000,
            max_block_ms=15000
        )

    return _producer


def validate_transaction(transaction):
    if not isinstance(transaction, dict) or not transaction:
        raise ValueError(
            "Transaction must be a non-empty JSON object."
        )

    required_features = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    missing_features = [
        feature
        for feature in required_features
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Missing transaction features: {', '.join(missing_features)}"
        )

    validated = {}

    for feature in required_features:
        value = transaction[feature]

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValueError(
                f"Feature '{feature}' must be numeric."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"Feature '{feature}' must be finite."
            )

        validated[feature] = float(value)

    if validated["Time"] < 0:
        raise ValueError("Time cannot be negative.")

    if validated["Amount"] < 0:
        raise ValueError("Amount cannot be negative.")

    return validated


def publish_transaction(transaction):
    validated_transaction = validate_transaction(transaction)

    event = {
        "event_id": str(uuid.uuid4()),
        "event_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "transaction": validated_transaction
    }

    try:
        producer = get_producer()

        future = producer.send(
            KAFKA_TOPIC,
            value=event
        )

        metadata = future.get(timeout=20)

        result = {
            "published": True,
            "event_id": event["event_id"],
            "topic": metadata.topic,
            "partition": metadata.partition,
            "offset": metadata.offset
        }

        logger.info(
            "Transaction published successfully: %s",
            result
        )

        return result

    except (KafkaError, TimeoutError, OSError) as exc:
        logger.exception("Failed to publish transaction.")

        raise RuntimeError(
            f"Failed to publish transaction: {exc}"
        ) from exc


def close_producer():
    global _producer

    if _producer is not None:
        producer = _producer
        _producer = None

        try:
            producer.flush(timeout=10)
        finally:
            producer.close(timeout=10)


def create_test_transaction(
    amount=149.62,
    time_value=10000.0
):
    return {
        "Time": time_value,
        **{
            f"V{i}": 0.0
            for i in range(1, 29)
        },
        "Amount": amount
    }


def main():
    parser = argparse.ArgumentParser(
        description="Publish a transaction to Kafka."
    )

    parser.add_argument(
        "--amount",
        type=float,
        default=149.62
    )

    parser.add_argument(
        "--time",
        type=float,
        default=10000.0
    )

    args = parser.parse_args()

    transaction = create_test_transaction(
        amount=args.amount,
        time_value=args.time
    )

    try:
        result = publish_transaction(transaction)
        print(json.dumps(result, indent=2))

    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1)

    finally:
        close_producer()


if __name__ == "__main__":
    main()