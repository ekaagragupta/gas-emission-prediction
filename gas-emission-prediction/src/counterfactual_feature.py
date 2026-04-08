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


class CounterfactualSimulator:

    def __init__(self, seq_len=30):

        self.seq_len = seq_len
        self.model = keras.models.load_model(MODEL_PATH, compile=False)

        self.df = pd.read_csv(DATA_PATH)

        self.feature_cols = [c for c in self.df.columns if c not in ("Date", "AQI")]

    def build_sequences(self):

        data = self.df[self.feature_cols].values

        X = []

        for i in range(self.seq_len, len(data)):
            X.append(data[i-self.seq_len:i])

        return np.array(X)

    def predict_aqi(self, X):

        preds = self.model.predict(X, verbose=0)

        return preds.flatten()

    def simulate_intervention(self, feature, change_percent):

        df_cf = self.df.copy()

        if feature not in df_cf.columns:
            raise ValueError("Feature not found")

        df_cf[feature] = df_cf[feature] * (1 + change_percent/100)

        data = df_cf[self.feature_cols].values

        X_cf = []

        for i in range(self.seq_len, len(data)):
            X_cf.append(data[i-self.seq_len:i])

        X_cf = np.array(X_cf)

        preds_cf = self.predict_aqi(X_cf)

        return preds_cf

    def run_counterfactual(self, feature="PM2.5", reduction=-20):

        print("Running counterfactual simulation...")

        X = self.build_sequences()

        baseline_pred = self.predict_aqi(X)

        cf_pred = self.simulate_intervention(feature, reduction)

        plt.figure(figsize=(10,5))

        plt.plot(baseline_pred[:200], label="Baseline AQI")

        plt.plot(cf_pred[:200], label="Counterfactual AQI")

        plt.title(f"Counterfactual Simulation ({feature} {reduction}%)")

        plt.xlabel("Time")

        plt.ylabel("Predicted AQI")

        plt.legend()

        path = os.path.join(OUTPUT_DIR, "counterfactual_simulation.png")

        plt.savefig(path)

        plt.show()

        print("Saved:", path)


def run_counterfactual():

    sim = CounterfactualSimulator()

    sim.run_counterfactual(feature="PM2.5", reduction=-20)


if __name__ == "__main__":

    run_counterfactual()