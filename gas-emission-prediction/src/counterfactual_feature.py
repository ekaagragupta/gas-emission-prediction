import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "aqi_lstm_model.h5")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "emissions_clean.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "counterfactual")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Counterfactual Simulator
# ─────────────────────────────────────────────
class CounterfactualSimulator:

    def __init__(self, seq_len=30):

        self.seq_len = seq_len

        print("Loading trained LSTM model...")
        self.model = keras.models.load_model(MODEL_PATH, compile=False)

        print("Loading processed dataset...")
        self.df = pd.read_csv(DATA_PATH)

        # Build column order exactly like training pipeline
        target_col = "AQI"
        feature_cols = [c for c in self.df.columns if c not in ("Date", "AQI")]

        self.all_cols = [target_col] + feature_cols

        print("Total features used:", len(self.all_cols))


    # ─────────────────────────────────────────
    # Build LSTM sequences
    # ─────────────────────────────────────────
    def build_sequences(self, df):

        data = df[self.all_cols].values

        X = []

        for i in range(self.seq_len, len(data)):
            X.append(data[i-self.seq_len:i])

        return np.array(X)


    # ─────────────────────────────────────────
    # Predict AQI
    # ─────────────────────────────────────────
    def predict_aqi(self, X):

        preds = self.model.predict(X, verbose=0)

        return preds.flatten()


    # ─────────────────────────────────────────
    # Counterfactual intervention
    # ─────────────────────────────────────────
    def simulate_intervention(self, feature, change_percent):

        print(f"Applying intervention: {feature} {change_percent}%")

        df_cf = self.df.copy()

        if feature not in df_cf.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset")

        # Modify pollutant level
        df_cf[feature] = df_cf[feature] * (1 + change_percent/100)

        X_cf = self.build_sequences(df_cf)

        preds_cf = self.predict_aqi(X_cf)

        return preds_cf


    # ─────────────────────────────────────────
    # Run simulation
    # ─────────────────────────────────────────
    def run_counterfactual(self, feature="PM2.5", reduction=-20):

        print("Running counterfactual simulation...")

        # Baseline prediction
        X = self.build_sequences(self.df)
        baseline_pred = self.predict_aqi(X)

        # Counterfactual prediction
        cf_pred = self.simulate_intervention(feature, reduction)

        # Plot results
        plt.figure(figsize=(10,5))

        plt.plot(baseline_pred[:200], label="Baseline AQI", color="blue")

        plt.plot(cf_pred[:200], label="Counterfactual AQI", color="red")

        plt.title(f"Counterfactual Simulation ({feature} {reduction}%)")

        plt.xlabel("Time")

        plt.ylabel("Predicted AQI")

        plt.legend()

        save_path = os.path.join(OUTPUT_DIR, "counterfactual_simulation.png")

        plt.savefig(save_path)

        plt.show()

        print("Graph saved at:", save_path)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def run_counterfactual():

    simulator = CounterfactualSimulator()

    simulator.run_counterfactual(
        feature="PM2.5",
        reduction=-20
    )


if __name__ == "__main__":

    run_counterfactual()