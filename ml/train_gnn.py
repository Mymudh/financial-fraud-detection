import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_gnn.pt"
METRICS_PATH = ARTIFACTS_DIR / "gnn_metrics.json"

DEVICE = torch.device("cpu")
THRESHOLD = 0.5

FEATURE_NAMES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
]

logger = logging.getLogger(__name__)


class FraudGraphSAGE(nn.Module):
    def __init__(self, input_features=8, hidden_features=32):
        super().__init__()
        self.conv1 = SAGEConv(input_features, hidden_features)
        self.conv2 = SAGEConv(hidden_features, hidden_features)
        self.output = nn.Linear(hidden_features, 2)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        return self.output(x)


def _load_checkpoint():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"GNN checkpoint not found: {MODEL_PATH}. "
            "Run python ml/train_gnn.py first."
        )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    required_keys = {
        "model_state_dict",
        "input_features",
        "hidden_features",
    }

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid GNN checkpoint format.")

    if not required_keys.issubset(checkpoint):
        raise ValueError(
            "GNN checkpoint is missing required model configuration."
        )

    input_features = int(checkpoint["input_features"])
    hidden_features = int(checkpoint["hidden_features"])

    if input_features != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} GNN input features, "
            f"but checkpoint expects {input_features}."
        )

    model = FraudGraphSAGE(
        input_features=input_features,
        hidden_features=hidden_features,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def _load_metrics():
    if not METRICS_PATH.is_file():
        return None

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Unable to read GNN evaluation metrics.")
        return None


try:
    GNN_MODEL = _load_checkpoint()
    GNN_LOAD_ERROR = None
    logger.info("GNN model loaded successfully.")
except Exception as exc:
    GNN_MODEL = None
    GNN_LOAD_ERROR = str(exc)
    logger.exception("GNN model could not be loaded.")


def get_gnn_status():
    metrics = _load_metrics()

    return {
        "available": GNN_MODEL is not None,
        "message": (
            "GNN model loaded successfully."
            if GNN_MODEL is not None
            else GNN_LOAD_ERROR
        ),
        "device": str(DEVICE),
        "input_features": len(FEATURE_NAMES),
        "metrics": metrics,
    }


def predict_graph(
    features,
    edge_index,
    transaction_node_index,
    threshold=THRESHOLD,
):
    if GNN_MODEL is None:
        raise RuntimeError(
            f"GNN model is unavailable: {GNN_LOAD_ERROR}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    try:
        feature_array = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Graph features must be numeric."
        ) from exc

    if feature_array.ndim != 2:
        raise ValueError(
            "Graph features must be a two-dimensional array."
        )

    if feature_array.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Each graph node must contain exactly "
            f"{len(FEATURE_NAMES)} features."
        )

    if feature_array.shape[0] == 0:
        raise ValueError("Graph must contain at least one node.")

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "Graph features must contain only finite values."
        )

    try:
        edges = np.asarray(edge_index, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Graph edges must contain integer node indices."
        ) from exc

    if edges.size == 0:
        edges = np.empty((2, 0), dtype=np.int64)

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, number_of_edges]."
        )

    num_nodes = feature_array.shape[0]

    if edges.size and (
        edges.min() < 0 or edges.max() >= num_nodes
    ):
        raise ValueError(
            "Graph contains an edge referencing an invalid node."
        )

    if isinstance(transaction_node_index, bool):
        raise ValueError("Transaction node index must be an integer.")

    try:
        node_index = int(transaction_node_index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Transaction node index must be an integer."
        ) from exc

    if node_index != transaction_node_index:
        raise ValueError(
            "Transaction node index must be an integer."
        )

    if not 0 <= node_index < num_nodes:
        raise ValueError(
            "Transaction node index is outside the graph."
        )

    if isinstance(threshold, bool):
        raise ValueError("Threshold must be numeric.")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Threshold must be numeric.") from exc

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    graph_data = Data(
        x=torch.tensor(feature_array, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long),
    ).to(DEVICE)

    GNN_MODEL.eval()

    with torch.no_grad():
        logits = GNN_MODEL(graph_data)
        probabilities = torch.softmax(logits, dim=1)
        fraud_probability = float(
            probabilities[node_index, 1].item()
        )

    prediction = int(fraud_probability >= threshold)

    return {
        "prediction": prediction,
        "label": (
            "Fraudulent" if prediction == 1 else "Legitimate"
        ),
        "fraud_probability": round(fraud_probability, 6),
        "threshold": threshold,
        "transaction_node_index": node_index,
        "graph_nodes": num_nodes,
        "graph_edges": int(edges.shape[1]),
        "model": "GraphSAGE",
    }


if __name__ == "__main__":
    print(json.dumps(get_gnn_status(), indent=4))