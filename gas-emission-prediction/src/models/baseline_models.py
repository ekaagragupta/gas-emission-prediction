

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# METRIC CALCULATION


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    """

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero
    mape = np.mean(
        np.abs((y_true - y_pred) / (y_true + 1e-8))
    ) * 100

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


# FLATTEN SEQUENCES FOR TABULAR MODELS


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Convert 3D LSTM input into 2D tabular format.

    (samples, sequence_length, features)
    → (samples, sequence_length * features)
    """
    return X.reshape(X.shape[0], -1)


#linear regression

def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LinearRegression:

    model = LinearRegression()
    model.fit(X_train, y_train)

    logger.info("Linear Regression trained.")
    return model

#random forest

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> RandomForestRegressor:

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    logger.info("Random Forest trained.")
    return model


#basic lstm model

def build_basic_lstm(
    input_shape: Tuple[int, int]
) -> tf.keras.Model:
    """
    Build simple 1-layer LSTM baseline.
    """

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    logger.info("Basic LSTM built.")
    return model


#main function to run all baseline experiments

def run_baseline_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Train and evaluate all baseline models.

    Returns:
        Sorted DataFrame of results.
    """

    results = []
# Flatten sequences for tabular models
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    X_test_flat = flatten_sequences(X_test)

    # ─────────────────────────────
    #  Linear Regression
    # ─────────────────────────────
    lr = train_linear_regression(X_train_flat, y_train)
    lr_pred = lr.predict(X_test_flat)

    lr_metrics = calculate_metrics(y_test, lr_pred)
    lr_metrics["Model"] = "Linear Regression"
    results.append(lr_metrics)

    # ─────────────────────────────
    # Random Forest
    # ─────────────────────────────
    rf = train_random_forest(X_train_flat, y_train)
    rf_pred = rf.predict(X_test_flat)

    rf_metrics = calculate_metrics(y_test, rf_pred)
    rf_metrics["Model"] = "Random Forest"
    results.append(rf_metrics)

    # ─────────────────────────────
    #  Basic LSTM
    # ─────────────────────────────
    basic_lstm = build_basic_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    basic_lstm.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        shuffle=False,  # IMPORTANT for time-series
        verbose=1
    )

    lstm_pred = basic_lstm.predict(X_test).flatten()

    lstm_metrics = calculate_metrics(y_test, lstm_pred)
    lstm_metrics["Model"] = "Basic LSTM"
    results.append(lstm_metrics)

  # Convert results to DataFrame and sort by R2
    df_results = pd.DataFrame(results)

    df_results = df_results[
        ["Model", "R2", "MAE", "RMSE", "MAPE"]
    ].sort_values("R2", ascending=False)

    logger.info("Baseline experiments completed.")
    return df_results