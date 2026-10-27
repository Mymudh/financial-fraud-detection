from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "datasets" / "creditcard.csv"
ARTIFACTS_PATH = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_PATH / "fraud_model.joblib"
METRICS_PATH = ARTIFACTS_PATH / "evaluation_metrics.json"
FEATURES_PATH = ARTIFACTS_PATH / "model_features.json"
MLFLOW_DB = ROOT / "mlflow.db"

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5

N_ESTIMATORS = 300
MAX_DEPTH = 5
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("Financial-Fraud-Detection")


def validate_dataset(df):
    if df.empty:
        raise ValueError("Dataset is empty.")

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in: {missing}")

    if not df.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if set(df["Class"].unique()) != {0, 1}:
        raise ValueError(
            "Class must contain both 0 (legitimate) and 1 (fraud)."
        )

    non_numeric = df.drop(columns=["Class"]).select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric model features found: {non_numeric}"
        )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if not X.columns.is_unique:
        raise ValueError("Model features must have unique names.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    fraud_count = int((y_train == 1).sum())
    legitimate_count = int((y_train == 0).sum())

    if fraud_count == 0:
        raise ValueError(
            "Training data contains no fraud transactions."
        )

    scale_pos_weight = legitimate_count / fraud_count

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    with mlflow.start_run(
        run_name="XGBoost-Fraud-Detection"
    ) as run:

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        metrics = {
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
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "pr_auc": float(
                average_precision_score(y_test, probabilities)
            ),
            "threshold": THRESHOLD,
        }

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            output_dict=True,
            zero_division=0,
        )

        params = {
            "model": "XGBClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "scale_pos_weight": scale_pos_weight,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "input_features": X.shape[1],
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "project": "Financial-Fraud-Detection",
            "algorithm": "XGBoost",
            "problem_type": "binary_classification",
        })

        confusion_data = {
            "labels": ["legitimate", "fraud"],
            "matrix": matrix.tolist(),
        }

        with (ARTIFACTS_PATH / "confusion_matrix.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(confusion_data, file, indent=4)

        with (ARTIFACTS_PATH / "classification_report.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=4)

        with FEATURES_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "feature_names": X.columns.tolist(),
                    "feature_count": len(X.columns),
                },
                file,
                indent=4,
            )

        evaluation_results = {
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
            "parameters": params,
        }

        with METRICS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                evaluation_results,
                file,
                indent=4,
            )

        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(FEATURES_PATH),
            artifact_path="model_metadata",
        )

        mlflow.log_artifact(
            str(ARTIFACTS_PATH / "confusion_matrix.json"),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(ARTIFACTS_PATH / "classification_report.json"),
            artifact_path="evaluation",
        )

        print("\n" + "=" * 55)
        print("MODEL TRAINING COMPLETED")
        print("=" * 55)

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Experiment: Financial-Fraud-Detection")

        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=4))

        print("\nConfusion matrix:")
        print(matrix)

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["legitimate", "fraud"],
                zero_division=0,
            )
        )

        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Metrics saved to: {METRICS_PATH}")
        print(f"Feature metadata saved to: {FEATURES_PATH}")
        print(f"MLflow database: {MLFLOW_DB}")


if __name__ == "__main__":
    main()