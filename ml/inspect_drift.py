from pathlib import Path
import json
import evidently
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "creditcard.csv"
OUTPUT = ROOT / "artifacts" / "evidently_debug.json"

print("Evidently version:", evidently.__version__)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

data = pd.read_csv(DATASET).dropna()

features = [column for column in data.columns if column != "Class"]

if len(data) < 15000:
    raise ValueError(f"At least 15000 rows are required; found {len(data)}")

reference = data.iloc[:10000][features].copy()
current = data.iloc[10000:15000][features].copy()

report = Report(metrics=[DataDriftPreset()])

result = report.run(
    reference_data=reference,
    current_data=current
)

report_dict = result.dict()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(report_dict, file, indent=2, default=str)

print("Reference rows:", len(reference))
print("Current rows:", len(current))
print("Feature count:", len(features))
print("Top-level report keys:", list(report_dict.keys()))
print("Debug report saved to:", OUTPUT)
print("\nReport preview:")
print(json.dumps(report_dict, indent=2, default=str)[:6000])