import torch
import pytest

from backend.gnn_predictor import predict_graph


def create_graph():
    torch.manual_seed(42)

    features = torch.rand(10, 8, dtype=torch.float32)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2],
            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 2, 1],
        ],
        dtype=torch.long,
    )

    return features, edge_index


def test_predict_graph_returns_result():
    features, edge_index = create_graph()

    result = predict_graph(
        features=features,
        edge_index=edge_index,
        transaction_node_index=0,
    )

    assert isinstance(result, dict)
    assert result


def test_predict_graph_has_valid_node_index():
    features, edge_index = create_graph()

    with pytest.raises((ValueError, IndexError, RuntimeError)):
        predict_graph(
            features=features,
            edge_index=edge_index,
            transaction_node_index=100,
        )


def test_predict_graph_rejects_invalid_features():
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        predict_graph(
            features=None,
            edge_index=None,
            transaction_node_index=0,
        )