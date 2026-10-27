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

logging.basicConfig(level=logging.INFO)
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


def load_gnn_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"GNN model not found: {MODEL_PATH}. "
            "Run python ml/train_gnn.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = FraudGraphSAGE(
        input_features=int(checkpoint["input_features"]),
        hidden_features=int(checkpoint["hidden_features"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


try:
    GNN_MODEL = load_gnn_model()
    GNN_LOAD_ERROR = None
    logger.info("GNN model loaded successfully.")
except Exception as exc:
    GNN_MODEL = None
    GNN_LOAD_ERROR = str(exc)
    logger.exception("GNN model loading failed.")


def get_gnn_status():
    metrics = None

    if METRICS_PATH.is_file():
        try:
            with METRICS_PATH.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not load GNN metrics.")

    return {
        "available": GNN_MODEL is not None,
        "message": (
            "GNN model loaded successfully."
            if GNN_MODEL is not None
            else GNN_LOAD_ERROR
        ),
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

    if isinstance(threshold, bool):
        raise ValueError("Threshold must be between 0 and 1.")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Threshold must be numeric.") from exc

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric.") from exc

    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(
            "Features must have shape [number_of_nodes, 8]."
        )

    if x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError(
            "Graph features must be non-empty and finite."
        )

    try:
        edges = np.asarray(edge_index, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Edge indices must be integers.") from exc

    if edges.size == 0:
        edges = np.empty((2, 0), dtype=np.int64)

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, number_of_edges]."
        )

    if edges.size and (
        edges.min() < 0 or edges.max() >= x.shape[0]
    ):
        raise ValueError("Graph contains invalid node indices.")

    if isinstance(transaction_node_index, bool):
        raise ValueError("Node index must be an integer.")

    try:
        node_index = int(transaction_node_index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Node index must be an integer.") from exc

    if node_index != transaction_node_index:
        raise ValueError("Node index must be an integer.")

    if not 0 <= node_index < x.shape[0]:
        raise ValueError("Node index is outside the graph.")

    graph = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long),
    ).to(DEVICE)

    GNN_MODEL.eval()

    with torch.no_grad():
        logits = GNN_MODEL(graph)
        probability = torch.softmax(logits, dim=1)[node_index, 1].item()

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": "Fraudulent" if prediction else "Legitimate",
        "fraud_probability": round(float(probability), 6),
        "threshold": threshold,
        "transaction_node_index": node_index,
        "graph_nodes": int(x.shape[0]),
        "graph_edges": int(edges.shape[1]),
        "model": "GraphSAGE",
    }


if __name__ == "__main__":
    print(json.dumps(get_gnn_status(), indent=4))