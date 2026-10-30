import pandas as pd

df = pd.read_csv("datasets/creditcard.csv")

normal = df[df["Class"] == 0].sample(
    n=min(2000, (df["Class"] == 0).sum()),
    random_state=42
)

fraud = df[df["Class"] == 1]

sample = pd.concat([normal, fraud])
sample = sample.sample(frac=1, random_state=42)

sample.to_csv("datasets/sample_creditcard.csv", index=False)

print("Sample dataset created successfully")
print("Shape:", sample.shape)
print(sample["Class"].value_counts())