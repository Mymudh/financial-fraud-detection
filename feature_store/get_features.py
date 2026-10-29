from pathlib import Path

import pandas as pd
from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE_DIR = PROJECT_ROOT / "feature_store"

store = FeatureStore(repo_path=str(FEATURE_STORE_DIR))


def get_transaction_features(transaction_id: int) -> dict:
    if transaction_id <= 0:
        raise ValueError("Transaction ID must be positive.")

    features = store.get_online_features(
        features=[
            "transaction_features:Amount",
            "transaction_features:Amount_Log",
            "transaction_features:Time_Sin",
            "transaction_features:Time_Cos",
            "transaction_features:Amount_Rank_Percentile",
            "transaction_features:Amount_ZScore",
        ],
        entity_rows=[
            {"transaction_id": transaction_id}
        ],
    ).to_dict()

    result = {}

    for name, values in features.items():
        if name == "transaction_id":
            continue

        value = values[0]

        if value is None or pd.isna(value):
            raise ValueError(
                f"Feature '{name}' is missing for transaction {transaction_id}."
            )

        result[name] = float(value)

    return result


if __name__ == "__main__":
    try:
        result = get_transaction_features(1)
        print("FEAST FEATURE RETRIEVAL SUCCESSFUL")
        print(result)

    except Exception as exc:
        print("FEATURE RETRIEVAL FAILED")
        print(f"{type(exc).__name__}: {exc}")