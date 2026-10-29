from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "creditcard.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("Financial-Fraud-Detection")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

if "Class" not in df.columns:
    raise ValueError("The dataset must contain a 'Class' column.")

if df["Class"].isnull().any():
    raise ValueError("The target column contains missing values.")

if df["Class"].nunique() != 2:
    raise ValueError("The target column must contain two classes.")

X = df.drop(columns=["Class"])
y = df["Class"]

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("The target column must contain only 0 and 1.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

params = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": 4,
}

model = XGBClassifier(**params)

print("=" * 60)
print("FINANCIAL FRAUD DETECTION MODEL TRAINING")
print("=" * 60)
print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Training features: {X_train.shape[1]}")
print(f"Fraudulent training transactions: {positive_count}")
print(f"Legitimate training transactions: {negative_count}")

with mlflow.start_run(run_name="xgboost-fraud-detection"):
    mlflow.log_params(params)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("split_random_state", 42)
    mlflow.log_param("classification_threshold", 0.5)
    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))
    mlflow.log_param("feature_count", X_train.shape[1])

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
        output_dict=True,
    )

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("CONFUSION MATRIX")
    print(cm)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    fraud_metrics = report["Fraudulent"]

    metrics_to_log = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fraud_precision": float(fraud_metrics["precision"]),
        "fraud_recall": float(fraud_metrics["recall"]),
        "fraud_f1_score": float(fraud_metrics["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_negatives": int(cm[0, 0]),
        "true_positives": int(cm[1, 1]),
    }

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    mlflow.log_metrics(metrics_to_log)

    model_path = ARTIFACTS_DIR / "fraud_model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(pr_auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(X_train)),
        "features": X.columns.tolist(),
    }

    metrics_path = ARTIFACTS_DIR / "evaluation_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    mlflow.log_artifact(
        str(metrics_path),
        artifact_path="evaluation",
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="model_files",
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="trained_model",
    )

    print("\nMODEL ARTIFACTS")
    print(f"Trained model: {model_path}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"MLflow database: {MLFLOW_DB}")
    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

print("\nModel training and MLflow logging completed successfully.")