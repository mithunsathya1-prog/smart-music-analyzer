# src/models/inspect_features.py
import pandas as pd

df = pd.read_csv("data/features/features.csv")

print("First few rows:\n", df.head(), "\n")
print("Unique labels (genres):", df['label'].unique(), "\n")
print("Count per label:\n", df['label'].value_counts(), "\n")
