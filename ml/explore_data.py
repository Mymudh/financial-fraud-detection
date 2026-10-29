from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "creditcard.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("FINANCIAL FRAUD DETECTION - DATA EXPLORATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

print("\nFraudulent transactions:")
print(int(df["Class"].sum()))

print("\nLegitimate transactions:")
print(int((df["Class"] == 0).sum()))

print("\nDataset exploration completed successfully.")