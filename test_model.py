from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
DATA_PATH = ROOT / "datasets" / "creditcard.csv"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH, nrows=5)

    X = data.drop(columns=["Class"])

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

        if list(X.columns) != expected:
            raise ValueError(
                "Dataset feature columns do not match the model."
            )

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nMODEL TEST SUCCESSFUL")
    print("=" * 40)
    print("Model:", type(model).__name__)
    print("Input feature count:", X.shape[1])

    for index, (prediction, probability) in enumerate(
        zip(predictions, probabilities),
        start=1,
    ):
        label = "Fraud" if prediction == 1 else "Legitimate"

        print(f"\nTransaction {index}")
        print("Prediction:", label)
        print(f"Fraud probability: {probability:.6f}")


if __name__ == "__main__":
    main()