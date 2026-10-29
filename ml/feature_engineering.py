from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "creditcard.csv"
FEATURES_DIR = PROJECT_ROOT / "artifacts" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FEATURES_DIR / "engineered_features.csv"
SUMMARY_PATH = FEATURES_DIR / "feature_summary.json"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

required_columns = {"Time", "Amount", "Class"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

if df[list(required_columns)].isnull().any().any():
    raise ValueError("Required columns contain missing values.")

if not set(df["Class"].unique()).issubset({0, 1}):
    raise ValueError("Class must contain only 0 and 1.")

if (df["Amount"] < 0).any():
    raise ValueError("Amount contains negative values; inspect the dataset.")

features = df.copy()

features["transaction_id"] = np.arange(1, len(features) + 1, dtype=np.int64)

features["event_timestamp"] = (
    pd.Timestamp("2026-01-01", tz="UTC")
    + pd.to_timedelta(features["Time"], unit="s")
)

features["Amount_Log"] = np.log1p(features["Amount"])

features["Transaction_Time_Seconds"] = features["Time"]

features["Time_Sin"] = np.sin(
    2 * np.pi * (features["Time"] % 86400) / 86400
)

features["Time_Cos"] = np.cos(
    2 * np.pi * (features["Time"] % 86400) / 86400
)

features["Amount_Rank_Percentile"] = features["Amount"].rank(
    method="average",
    pct=True,
)

amount_std = features["Amount"].std(ddof=0)

if amount_std == 0:
    features["Amount_ZScore"] = 0.0
else:
    features["Amount_ZScore"] = (
        features["Amount"] - features["Amount"].mean()
    ) / amount_std

features.replace([np.inf, -np.inf], np.nan, inplace=True)

if features.isnull().any().any():
    raise ValueError("Generated features contain missing or infinite values.")

features.to_csv(OUTPUT_PATH, index=False)

summary = {
    "source_dataset": str(DATASET_PATH),
    "output_dataset": str(OUTPUT_PATH),
    "rows": int(len(features)),
    "columns": int(len(features.columns)),
    "new_features": [
        "transaction_id",
        "event_timestamp",
        "Amount_Log",
        "Transaction_Time_Seconds",
        "Time_Sin",
        "Time_Cos",
        "Amount_Rank_Percentile",
        "Amount_ZScore",
    ],
    "fraudulent_transactions": int((features["Class"] == 1).sum()),
    "legitimate_transactions": int((features["Class"] == 0).sum()),
    "missing_values": int(features.isnull().sum().sum()),
    "timestamp_note": (
        "Synthetic UTC timestamps based on the dataset's elapsed Time column; "
        "not actual transaction timestamps."
    ),
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)
print(f"Total rows: {len(features)}")
print(f"Total columns: {len(features.columns)}")
print(f"Missing values: {features.isnull().sum().sum()}")
print(f"Feature dataset: {OUTPUT_PATH}")
print(f"Feature summary: {SUMMARY_PATH}")
print("\nFeature engineering completed successfully.")