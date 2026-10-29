
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "creditcard.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "fraud_gnn.pt"
METRICS_PATH = ARTIFACTS / "gnn_metrics.json"

SEED = 42
HIDDEN_FEATURES = 32
EPOCHS = 20
DEVICE = torch.device("cpu")

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


def build_graph(features):
    n = len(features)

    source = np.arange(n - 1, dtype=np.int64)
    target = source + 1

    edge_index = np.vstack([
        np.concatenate([source, target]),
        np.concatenate([target, source]),
    ])

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not DATASET.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain the Class target column.")

    feature_columns = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
    ]

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).to_numpy(dtype=np.float32)

    y = df["Class"].to_numpy(dtype=np.int64)

    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_graph = build_graph(X_train)
    test_graph = build_graph(X_test)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = FraudGraphSAGE(
        input_features=8,
        hidden_features=HIDDEN_FEATURES,
    ).to(DEVICE)

    train_graph = train_graph.to(DEVICE)
    test_graph = test_graph.to(DEVICE)
    y_train_tensor = y_train_tensor.to(DEVICE)
    y_test_tensor = y_test_tensor.to(DEVICE)

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (
        2.0 * np.maximum(class_counts, 1)
    )

    weights = torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.003, weight_decay=1e-4
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        logits = model(train_graph)
        loss = F.cross_entropy(
            logits, y_train_tensor, weight=weights
        )

        loss.backward()
        optimizer.step()

        logger.info(
            "Epoch %d/%d | Loss: %.5f",
            epoch + 1,
            EPOCHS,
            loss.item(),
        )

    model.eval()
    with torch.no_grad():
        logits = model(test_graph)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    predictions = (probabilities >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "test_samples": int(len(y_test)),
        "epochs": EPOCHS,
        "threshold": 0.5,
        "features": feature_columns,
        "note": (
            "Experimental GraphSAGE baseline. Graph edges connect "
            "adjacent rows and do not represent verified transaction "
            "relationships. Evaluate graph construction before "
            "using this model for real fraud decisions."
        ),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 8,
            "hidden_features": HIDDEN_FEATURES,
            "feature_names": [
                f"feature_{i}" for i in range(1, 9)
            ],
            "threshold": 0.5,
        },
        MODEL_PATH,
    )

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved metrics to %s", METRICS_PATH)
    logger.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
