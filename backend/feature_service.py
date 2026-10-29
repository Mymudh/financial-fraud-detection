from pathlib import Path

import pandas as pd
from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE_PATH = PROJECT_ROOT / "feature_store"

FEATURE_REFERENCES = [
    "transaction_features:Amount",
    "transaction_features:Amount_Log",
    "transaction_features:Amount_ZScore",
    "transaction_features:Amount_Rank_Percentile",
    "transaction_features:Time_Sin",
    "transaction_features:Time_Cos",
]


def get_transaction_features(transaction_id: int) -> dict:
    if transaction_id <= 0:
        raise ValueError("Transaction ID must be a positive integer.")

    if not FEATURE_STORE_PATH.exists():
        raise RuntimeError(
            f"Feast feature store directory not found: {FEATURE_STORE_PATH}"
        )

    try:
        store = FeatureStore(repo_path=str(FEATURE_STORE_PATH))

        result = store.get_online_features(
            features=FEATURE_REFERENCES,
            entity_rows=[{"transaction_id": transaction_id}],
        ).to_dict()

    except Exception as exc:
        raise RuntimeError(
            f"Unable to retrieve Feast features: {exc}"
        ) from exc

    features = {}

    for name, values in result.items():
        if name == "transaction_id":
            continue

        value = values[0] if values else None

        if value is None or pd.isna(value):
            raise ValueError(
                f"Feature '{name}' was not found for transaction "
                f"{transaction_id}."
            )

        features[name] = float(value)

    missing_features = set(FEATURE_REFERENCES) - {
        f"transaction_features:{name}" for name in features
    }

    if missing_features:
        raise ValueError(
            f"Missing Feast features: {sorted(missing_features)}"
        )

    return features