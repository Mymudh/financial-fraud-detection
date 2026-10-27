from pathlib import Path

from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = PROJECT_ROOT / "feature_store"


def test_feast_online_features():
    store = FeatureStore(repo_path=str(STORE_DIR))

    result = store.get_online_features(
        features=[
            "transaction_features:Amount",
            "transaction_features:Amount_Log",
            "transaction_features:Amount_ZScore",
            "transaction_features:Amount_Rank_Percentile",
            "transaction_features:Time_Sin",
            "transaction_features:Time_Cos",
        ],
        entity_rows=[{"transaction_id": 1}],
    ).to_dict()

    assert result["transaction_id"][0] == 1

    expected_features = [
        "Amount",
        "Amount_Log",
        "Amount_ZScore",
        "Amount_Rank_Percentile",
        "Time_Sin",
        "Time_Cos",
    ]

    for feature in expected_features:
        assert feature in result
        assert result[feature][0] is not None