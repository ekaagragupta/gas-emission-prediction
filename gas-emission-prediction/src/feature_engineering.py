# src/feature_engineering.py

import numpy as np
import pandas as pd
from typing import Tuple


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])

    df["hour"] = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_year"] = df["Date"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 7, 14, 30]:
       df[f"AQI_lag_{lag}"] = df["AQI"].shift(lag)
       df[f"NO2_lag_{lag}"] = df["NO2"].shift(lag)
    return df


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    """

    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])

    return np.array(X), np.array(y)
