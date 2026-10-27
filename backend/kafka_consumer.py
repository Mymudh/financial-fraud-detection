import json
import logging
import math
import os
from datetime import datetime, timezone

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from kafka.structs import OffsetAndMetadata, TopicPartition

from backend.predictor import predict_transaction


BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "fraud-transactions",
)

GROUP_ID = os.getenv(
    "KAFKA_GROUP_ID",
    "fraud-detection-consumer-v2",
)

POLL_TIMEOUT_MS = 2000
MAX_RECORDS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)
logging.getLogger("kafka").setLevel(logging.WARNING)


def deserialize_message(value):
    if value is None:
        raise ValueError("Kafka message value is empty.")

    try:
        decoded = value.decode("utf-8")
        event = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Kafka message contains invalid JSON: {exc}"
        ) from exc

    if not isinstance(event, dict):
        raise ValueError("Kafka message must contain a JSON object.")

    return event


def process_transaction(event):
    if not isinstance(event, dict):
        raise ValueError("Event must be a JSON object.")

    transaction = event.get("transaction", event)

    if not isinstance(transaction, dict):
        raise ValueError("Transaction must be a JSON object.")

    result = predict_transaction(transaction)

    required_keys = {
        "prediction",
        "label",
        "fraud_probability",
        "threshold",
    }

    missing_keys = required_keys.difference(result)

    if missing_keys:
        raise ValueError(
            f"Prediction result is missing keys: {sorted(missing_keys)}"
        )

    try:
        prediction = int(result["prediction"])
        probability = float(result["fraud_probability"])
        threshold = float(result["threshold"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Prediction, probability, and threshold must be numeric."
        ) from exc

    if prediction not in (0, 1):
        raise ValueError("Prediction must be 0 or 1.")

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "Fraud probability must be between 0 and 1."
        )

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    return {
        "event_id": event.get("event_id"),
        "transaction_id": transaction.get("transaction_id"),
        "prediction": prediction,
        "label": str(result["label"]),
        "fraud_probability": round(probability, 6),
        "threshold": threshold,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def commit_message(consumer, message):
    partition = TopicPartition(
        message.topic,
        message.partition,
    )

    consumer.commit(
        offsets={
            partition: OffsetAndMetadata(
                message.offset + 1,
                "",
            )
        }
    )


def handle_message(consumer, message):
    logger.info(
        "Received message | topic=%s | partition=%s | offset=%s",
        message.topic,
        message.partition,
        message.offset,
    )

    try:
        output = process_transaction(message.value)

        if output["prediction"] == 1:
            logger.warning(
                "FRAUD ALERT | transaction_id=%s | probability=%.6f",
                output["transaction_id"],
                output["fraud_probability"],
            )
        else:
            logger.info(
                "LEGITIMATE TRANSACTION | transaction_id=%s | "
                "probability=%.6f",
                output["transaction_id"],
                output["fraud_probability"],
            )

        print(json.dumps(output, indent=4), flush=True)

        commit_message(consumer, message)

        logger.info(
            "Message processed and offset committed: %s",
            message.offset,
        )

    except (ValueError, TypeError, KeyError) as exc:
        logger.exception(
            "Invalid transaction at offset %s: %s",
            message.offset,
            exc,
        )

        try:
            commit_message(consumer, message)
            logger.warning(
                "Invalid message skipped; offset committed: %s",
                message.offset,
            )
        except KafkaError:
            logger.exception(
                "Failed to commit invalid message offset %s.",
                message.offset,
            )

    except KafkaError:
        logger.exception(
            "Kafka error while processing offset %s.",
            message.offset,
        )

    except Exception:
        logger.exception(
            "Unexpected processing error at offset %s. "
            "The message was not committed.",
            message.offset,
        )


def main():
    consumer = None

    try:
        consumer = KafkaConsumer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id=GROUP_ID,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=deserialize_message,
            request_timeout_ms=40000,
            session_timeout_ms=10000,
            max_poll_records=MAX_RECORDS,
        )

        consumer.subscribe([TOPIC])

        logger.info("Fraud detection Kafka consumer started.")
        logger.info("Broker: %s", BOOTSTRAP_SERVERS)
        logger.info("Topic: %s", TOPIC)
        logger.info("Consumer group: %s", GROUP_ID)

        while True:
            records = consumer.poll(
                timeout_ms=POLL_TIMEOUT_MS,
                max_records=MAX_RECORDS,
            )

            if not records:
                logger.info("Waiting for transaction messages...")
                continue

            for messages in records.values():
                for message in messages:
                    handle_message(consumer, message)

    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")

    except KafkaError:
        logger.exception("Kafka connection or consumer error.")

    except Exception:
        logger.exception("Unexpected consumer error.")

    finally:
        if consumer is not None:
            consumer.close()
            logger.info("Kafka consumer closed.")


if __name__ == "__main__":
    main()