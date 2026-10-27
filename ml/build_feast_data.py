from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "creditcard.csv"
OUTPUT = ROOT / "feature_store" / "data" / "transaction_features.parquet"

df = pd.read_csv(DATASET)

required = {"Time", "Amount"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing dataset columns: {sorted(missing)}")

amount = df["Amount"].astype(float)
time = df["Time"].astype(float)
std = amount.std()

features = pd.DataFrame({
    "transaction_id": np.arange(1, len(df) + 1, dtype=np.int64),
    "event_timestamp": pd.Timestamp.now(tz="UTC"),
    "Amount": amount,
    "Amount_Log": np.log1p(amount.clip(lower=0)),
    "Amount_ZScore": (amount - amount.mean()) / (std if std > 0 else 1.0),
    "Amount_Rank_Percentile": amount.rank(pct=True),
    "Time_Sin": np.sin(2 * np.pi * time / 86400),
    "Time_Cos": np.cos(2 * np.pi * time / 86400),
})

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
features.to_parquet(OUTPUT, index=False)

print(f"Saved {len(features)} transactions to {OUTPUT}")
print(f"Columns: {list(features.columns)}")