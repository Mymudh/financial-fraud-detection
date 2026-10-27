from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "engineered_features.csv"
)


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found: {FEATURES_PATH}"
        )

    data = pd.read_csv(FEATURES_PATH)

    print("\nENGINEERED FEATURE VALIDATION")
    print("=" * 45)
    print(f"Dataset path: {FEATURES_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Missing values: {data.isna().sum().sum()}")
    print(f"Duplicate rows: {data.duplicated().sum()}")

    print("\nAll columns:")
    for column in data.columns:
        print(column)

    original_columns = {
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "Class",
    }

    new_columns = [
        column
        for column in data.columns
        if column not in original_columns
    ]

    print("\nEngineered columns:")
    for column in new_columns:
        print(column)

    if data.empty:
        raise ValueError("The engineered dataset is empty.")

    if "Class" not in data.columns:
        raise ValueError("The target column 'Class' is missing.")

    print("\nFeature validation completed successfully.")


if __name__ == "__main__":
    main()