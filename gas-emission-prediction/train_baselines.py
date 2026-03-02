import pandas as pd
import numpy as np

from src.feature_engineering import create_sequences
from src.models.baseline_models import run_baseline_experiments

def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Time-series safe split (no shuffling).
    """

    total = len(X)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

data = pd.read_csv("data/processed/emissions_clean.csv")


if "timestamp" in data.columns:
    data = data.drop(columns=["timestamp"])


data_values = data.values

X, y = create_sequences(data_values, sequence_length=30)


X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

results = run_baseline_experiments(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

print("\nBaseline Results:")
print(results)