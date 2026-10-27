import pytest
from unittest.mock import MagicMock, patch

from kafka.errors import KafkaError

from backend.kafka_producer import (
    publish_transaction,
    validate_transaction,
)


@pytest.fixture
def mock_producer():
    with patch("backend.kafka_producer.get_producer") as mock_get:
        producer = MagicMock()

        metadata = MagicMock()
        metadata.topic = "fraud-transactions"
        metadata.partition = 0
        metadata.offset = 1

        producer.send.return_value.get.return_value = metadata
        mock_get.return_value = producer

        yield producer


def create_transaction(amount=149.62, time_value=10000.0):
    return {
        "Time": time_value,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": amount
    }


def test_publish_transaction_success(mock_producer):
    result = publish_transaction(create_transaction())

    assert result["published"] is True
    assert result["topic"] == "fraud-transactions"
    assert result["partition"] == 0
    assert result["offset"] == 1
    assert result["event_id"]

    mock_producer.send.assert_called_once()
    mock_producer.flush.assert_not_called()
    mock_producer.close.assert_not_called()


def test_publish_transaction_contains_all_features(mock_producer):
    publish_transaction(create_transaction(amount=250.0))

    event = mock_producer.send.call_args.kwargs["value"]
    transaction = event["transaction"]

    assert transaction["Time"] == 10000.0
    assert transaction["Amount"] == 250.0

    for i in range(1, 29):
        assert f"V{i}" in transaction


@pytest.mark.parametrize(
    "invalid_input",
    ["invalid", [], None, 123, {}]
)
def test_publish_transaction_rejects_invalid_input(invalid_input):
    with pytest.raises(ValueError):
        validate_transaction(invalid_input)


def test_publish_transaction_rejects_missing_features():
    with pytest.raises(ValueError, match="Missing transaction features"):
        validate_transaction({
            "Time": 10000.0,
            "Amount": 100.0
        })


def test_publish_transaction_handles_kafka_failure(mock_producer):
    mock_producer.send.return_value.get.side_effect = KafkaError(
        "Kafka unavailable"
    )

    with pytest.raises(RuntimeError, match="Failed to publish transaction"):
        publish_transaction(create_transaction())

    mock_producer.send.assert_called_once()


def test_close_producer():
    with patch("backend.kafka_producer.get_producer") as mock_get:
        from backend import kafka_producer

        producer = MagicMock()
        mock_get.return_value = producer

        with patch.object(kafka_producer, "_producer", producer):
            kafka_producer.close_producer()

        producer.flush.assert_called_once_with(timeout=10)
        producer.close.assert_called_once_with(timeout=10)