import logging
import math
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "fraud_model.joblib"
FEATURE_STORE_PATH = PROJECT_ROOT / "feature_store"

THRESHOLD = 0.5

FEAST_FEATURES = [
    "transaction_features:Amount",
    "transaction_features:Amount_Log",
    "transaction_features:Time_Sin",
    "transaction_features:Time_Cos",
    "transaction_features:Amount_Rank_Percentile",
    "transaction_features:Amount_ZScore",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the model training script first."
        )

    loaded_model = joblib.load(MODEL_PATH)

    if not hasattr(loaded_model, "feature_names_in_"):
        raise ValueError(
            "The model does not contain feature names. "
            "Train the model using a pandas DataFrame."
        )

    if not hasattr(loaded_model, "predict_proba"):
        raise ValueError(
            "The trained model must support predict_proba()."
        )

    if not hasattr(loaded_model, "classes_"):
        raise ValueError(
            "The trained model does not expose its class labels."
        )

    feature_names = [
        str(name) for name in loaded_model.feature_names_in_
    ]

    if not feature_names:
        raise ValueError("The trained model contains no features.")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "The trained model contains duplicate feature names."
        )

    if 1 not in list(loaded_model.classes_):
        raise ValueError(
            "The trained model has no class 1 for fraud detection."
        )

    logger.info("Fraud detection model loaded successfully.")
    logger.info("Model features: %s", len(feature_names))

    return loaded_model, feature_names


def load_feature_store():
    if not FEATURE_STORE_PATH.is_dir():
        logger.warning(
            "Feast repository not found at %s.",
            FEATURE_STORE_PATH,
        )
        return None

    try:
        store = FeatureStore(
            repo_path=str(FEATURE_STORE_PATH)
        )
        logger.info("Feast feature store initialized.")
        return store

    except Exception:
        logger.exception("Feast initialization failed.")
        return None


model, FEATURE_NAMES = load_model()
FEATURE_STORE = load_feature_store()


def get_stored_features(transaction_id):
    if transaction_id is None:
        return {
            "available": False,
            "features": {},
            "message": "No transaction ID supplied.",
        }

    try:
        transaction_id = int(transaction_id)

        if transaction_id <= 0:
            raise ValueError

    except (TypeError, ValueError, OverflowError):
        return {
            "available": False,
            "features": {},
            "message": (
                "Transaction ID must be a positive integer."
            ),
        }

    if FEATURE_STORE is None:
        return {
            "available": False,
            "features": {},
            "message": "Feast feature store is unavailable.",
        }

    try:
        response = FEATURE_STORE.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[
                {"transaction_id": transaction_id}
            ],
        ).to_dict()

        stored_features = {}

        for feature_reference in FEAST_FEATURES:
            feature_name = feature_reference.split(":", 1)[1]

            if feature_name not in response:
                return {
                    "available": False,
                    "features": {},
                    "message": (
                        f"Feature '{feature_name}' was not returned "
                        "by Feast."
                    ),
                }

            values = response[feature_name]

            if not values:
                return {
                    "available": False,
                    "features": {},
                    "message": (
                        f"No value returned for '{feature_name}'."
                    ),
                }

            value = values[0]

            if value is None or pd.isna(value):
                return {
                    "available": False,
                    "features": {},
                    "message": (
                        f"Feature '{feature_name}' is missing "
                        f"for transaction {transaction_id}."
                    ),
                }

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                return {
                    "available": False,
                    "features": {},
                    "message": (
                        f"Feature '{feature_name}' is not finite."
                    ),
                }

            stored_features[feature_name] = numeric_value

        return {
            "available": True,
            "features": stored_features,
            "message": "Feast features retrieved successfully.",
        }

    except Exception as exc:
        logger.warning(
            "Feast feature retrieval failed for transaction %s: %s",
            transaction_id,
            exc,
        )

        return {
            "available": False,
            "features": {},
            "message": "Feast feature retrieval failed.",
        }


def predict_transaction(transaction_data):
    if not isinstance(transaction_data, dict):
        raise ValueError(
            "Transaction data must be a JSON object."
        )

    transaction_id = transaction_data.get("transaction_id")

    feast_result = get_stored_features(transaction_id)

    input_values = dict(transaction_data)

    feast_used = []

    if feast_result["available"]:
        for feature_name, value in feast_result["features"].items():
            if feature_name in FEATURE_NAMES:
                input_values[feature_name] = value
                feast_used.append(feature_name)

    missing_features = [
        name
        for name in FEATURE_NAMES
        if name not in input_values
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}. "
            "Provide the model's required input features or retrain "
            "the model using the engineered Feast features."
        )

    values = {}

    for feature_name in FEATURE_NAMES:
        raw_value = input_values[feature_name]

        try:
            numeric_value = float(raw_value)

        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Feature '{feature_name}' must be numeric."
            ) from exc

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Feature '{feature_name}' must be finite."
            )

        values[feature_name] = numeric_value

    input_df = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES,
    )

    try:
        probabilities = model.predict_proba(input_df)

        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "The model did not return binary-class probabilities."
            )

        classes = list(model.classes_)
        fraud_class_index = classes.index(1)

        probability = float(
            probabilities[0, fraud_class_index]
        )

    except Exception as exc:
        logger.exception("Fraud prediction failed.")

        raise RuntimeError(
            "The trained model could not generate a valid prediction."
        ) from exc

    if (
        not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise RuntimeError(
            "The model returned an invalid fraud probability."
        )

    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "label": (
            "Fraudulent" if prediction == 1 else "Legitimate"
        ),
        "fraud_probability": round(probability, 6),
        "threshold": THRESHOLD,
        "transaction_id": transaction_id,
        "feast_features": feast_result,
        "feast_features_used_for_prediction": feast_used,
        "model_features": FEATURE_NAMES,
    }


if __name__ == "__main__":
    logger.info("Predictor module loaded.")
    logger.info("Expected model features: %s", FEATURE_NAMES)

