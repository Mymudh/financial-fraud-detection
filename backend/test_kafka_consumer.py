import json
from unittest.mock import MagicMock, patch

import pytest
from kafka.structs import OffsetAndMetadata, TopicPartition

from backend import kafka_consumer


@pytest.fixture
def valid_transaction():
    return {
        "transaction_id": "txn-test-001",
        "Time": 1000.0,
        "V1": -1.2,
        "V2": 0.5,
        "Amount": 125.50,
    }


@pytest.fixture
def valid_prediction():
    return {
        "prediction": 1,
        "label": "fraud",
        "fraud_probability": 0.97,
        "threshold": 0.5,
    }


@pytest.fixture
def valid_event(valid_transaction):
    return {
        "event_id": "event-001",
        "transaction": valid_transaction,
    }


def test_deserialize_valid_message(valid_event):
    result = kafka_consumer.deserialize_message(
        json.dumps(valid_event).encode("utf-8")
    )

    assert result == valid_event


@pytest.mark.parametrize(
    "value",
    [
        None,
        b"\xff",
        b"{invalid-json}",
        b"[]",
        b'"string"',
        b"123",
    ],
)
def test_deserialize_invalid_message(value):
    with pytest.raises(ValueError):
        kafka_consumer.deserialize_message(value)


def test_process_transaction_success(
    valid_event,
    valid_prediction,
):
    with patch(
        "backend.kafka_consumer.predict_transaction",
        return_value=valid_prediction,
    ) as predictor:
        result = kafka_consumer.process_transaction(valid_event)

    predictor.assert_called_once_with(valid_event["transaction"])

    assert result["event_id"] == "event-001"
    assert result["transaction_id"] == "txn-test-001"
    assert result["prediction"] == 1
    assert result["label"] == "fraud"
    assert result["fraud_probability"] == 0.97
    assert result["threshold"] == 0.5
    assert result["processed_at"]


def test_process_transaction_accepts_direct_transaction(
    valid_transaction,
    valid_prediction,
):
    with patch(
        "backend.kafka_consumer.predict_transaction",
        return_value=valid_prediction,
    ) as predictor:
        result = kafka_consumer.process_transaction(
            valid_transaction
        )

    predictor.assert_called_once_with(valid_transaction)
    assert result["transaction_id"] == "txn-test-001"


@pytest.mark.parametrize(
    "event",
    [
        None,
        [],
        "invalid",
        {"transaction": None},
        {"transaction": []},
    ],
)
def test_process_transaction_rejects_invalid_event(event):
    with pytest.raises(ValueError):
        kafka_consumer.process_transaction(event)


@pytest.mark.parametrize(
    "prediction",
    [
        {},
        {
            "prediction": 1,
            "label": "fraud",
            "threshold": 0.5,
        },
        {
            "prediction": 1,
            "label": "fraud",
            "fraud_probability": 0.8,
        },
    ],
)
def test_process_transaction_rejects_missing_prediction_fields(
    valid_transaction,
    prediction,
):
    with patch(
        "backend.kafka_consumer.predict_transaction",
        return_value=prediction,
    ):
        with pytest.raises(ValueError):
            kafka_consumer.process_transaction(
                {"transaction": valid_transaction}
            )


@pytest.mark.parametrize(
    "prediction",
    [
        2,
        -1,
        "invalid",
        None,
    ],
)
def test_process_transaction_rejects_invalid_classification(
    valid_transaction,
    prediction,
):
    result = {
        "prediction": prediction,
        "label": "fraud",
        "fraud_probability": 0.8,
        "threshold": 0.5,
    }

    with patch(
        "backend.kafka_consumer.predict_transaction",
        return_value=result,
    ):
        with pytest.raises(ValueError):
            kafka_consumer.process_transaction(
                {"transaction": valid_transaction}
            )


@pytest.mark.parametrize(
    "probability",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "invalid",
        None,
    ],
)
def test_process_transaction_rejects_invalid_probability(
    valid_transaction,
    probability,
):
    result = {
        "prediction": 1,
        "label": "fraud",
        "fraud_probability": probability,
        "threshold": 0.5,
    }

    with patch(
        "backend.kafka_consumer.predict_transaction",
        return_value=result,
    ):
        with pytest.raises(ValueError):
            kafka_consumer.process_transaction(
                {"transaction": valid_transaction}
            )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "invalid",
        None,
    ],
)
def test_process_transaction_rejects_invalid_threshold(
    valid_transaction,
    threshold,
):
    result = {
        "prediction": 1,
        "label": "fraud",
        "fraud_probability": 0.8,
        "threshold": threshold,
    }

    with patch(
        "backend.kafka_consumer.predict_transaction",
        return_value=result,
    ):
        with pytest.raises(ValueError):
            kafka_consumer.process_transaction(
                {"transaction": valid_transaction}
            )


def test_commit_message_uses_next_offset():
    consumer = MagicMock()

    message = MagicMock()
    message.topic = "fraud-transactions"
    message.partition = 2
    message.offset = 41

    kafka_consumer.commit_message(consumer, message)

    consumer.commit.assert_called_once()

    offsets = consumer.commit.call_args.kwargs["offsets"]

    partition = TopicPartition(
        "fraud-transactions",
        2,
    )

    assert offsets[partition] == OffsetAndMetadata(42, "")


def test_handle_message_processes_and_commits(
    valid_transaction,
    valid_prediction,
):
    consumer = MagicMock()

    message = MagicMock()
    message.topic = "fraud-transactions"
    message.partition = 0
    message.offset = 10
    message.value = {
        "event_id": "event-001",
        "transaction": valid_transaction,
    }

    prediction_result = {
        "event_id": "event-001",
        "transaction_id": "txn-test-001",
        **valid_prediction,
        "processed_at": "2026-01-01T00:00:00+00:00",
    }

    with patch(
        "backend.kafka_consumer.process_transaction",
        return_value=prediction_result,
    ), patch("builtins.print") as mock_print:
        kafka_consumer.handle_message(consumer, message)

    consumer.commit.assert_called_once()
    mock_print.assert_called_once()


def test_handle_message_skips_invalid_transaction():
    consumer = MagicMock()

    message = MagicMock()
    message.topic = "fraud-transactions"
    message.partition = 0
    message.offset = 11
    message.value = {"transaction": {}}

    with patch(
        "backend.kafka_consumer.process_transaction",
        side_effect=ValueError("Invalid transaction"),
    ):
        kafka_consumer.handle_message(consumer, message)

    consumer.commit.assert_called_once()


def test_handle_message_does_not_commit_unexpected_error():
    consumer = MagicMock()

    message = MagicMock()
    message.topic = "fraud-transactions"
    message.partition = 0
    message.offset = 12
    message.value = {"transaction": {}}

    with patch(
        "backend.kafka_consumer.process_transaction",
        side_effect=RuntimeError("Unexpected failure"),
    ):
        kafka_consumer.handle_message(consumer, message)

    consumer.commit.assert_not_called()


def test_handle_message_does_not_commit_when_commit_fails(
    valid_transaction,
    valid_prediction,
):
    consumer = MagicMock()

    message = MagicMock()
    message.topic = "fraud-transactions"
    message.partition = 0
    message.offset = 13
    message.value = {
        "event_id": "event-001",
        "transaction": valid_transaction,
    }

    consumer.commit.side_effect = RuntimeError("Commit failed")

    with patch(
        "backend.kafka_consumer.process_transaction",
        return_value={
            "event_id": "event-001",
            "transaction_id": "txn-test-001",
            **valid_prediction,
            "processed_at": "2026-01-01T00:00:00+00:00",
        },
    ), patch("builtins.print"):
        kafka_consumer.handle_message(consumer, message)

    consumer.commit.assert_called_once()