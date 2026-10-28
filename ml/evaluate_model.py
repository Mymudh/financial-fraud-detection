import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "creditcard.csv"
MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
OUTPUT_PATH = ROOT / "artifacts" / "evaluation_report.json"


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    data = pd.read_csv(DATASET)

    if "Class" not in data.columns:
        raise ValueError(
            "Dataset must contain the Class target column."
        )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

        if set(expected) != set(X_test.columns):
            raise ValueError(
                "Dataset columns do not match the trained model features."
            )

        X_test = X_test[expected]

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)
    else:
        probabilities = None

    report = {
        "test_samples": int(len(y_test)),
        "fraud_samples": int(y_test.sum()),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if probabilities is not None:
        report["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )
        report["average_precision"] = float(
            average_precision_score(y_test, probabilities)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
