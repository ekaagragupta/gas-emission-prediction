import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "aqi_lstm_model.h5")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "emissions_clean.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "uncertainty")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class UncertaintyEstimator:

    def __init__(self, seq_len=30):

        print("Loading trained model...")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print("Loading dataset...")
        self.df = pd.read_csv(DATA_PATH)

        if "Date" in self.df.columns:
            self.df = self.df.drop(columns=["Date"])

        self.seq_len = seq_len

        print("Dataset shape:", self.df.shape)


    # ------------------------------
    # Create LSTM sequences
    # ------------------------------
    def build_sequences(self):

        data = self.df.values

        X = []

        for i in range(self.seq_len, len(data)):
            X.append(data[i-self.seq_len:i])

        return np.array(X)


    # Monte Carlo Dropout Prediction

    def mc_dropout_prediction(self, X, n_samples=50):

        predictions = []

        for _ in range(n_samples):

            preds = self.model(X, training=True).numpy().flatten()

            predictions.append(preds)

        predictions = np.array(predictions)

        mean_pred = predictions.mean(axis=0)

        std_pred = predictions.std(axis=0)

        return mean_pred, std_pred


    # ------------------------------
    # Plot uncertainty
    # ------------------------------
    def plot_uncertainty(self, mean_pred, std_pred):

        time = np.arange(len(mean_pred))

        plt.figure(figsize=(10,5))

        plt.plot(time[:200], mean_pred[:200], label="Predicted AQI")

        lower = mean_pred - 2*std_pred
        upper = mean_pred + 2*std_pred

        plt.fill_between(
            time[:200],
            lower[:200],
            upper[:200],
            alpha=0.3,
            label="Confidence Interval"
        )

        plt.xlabel("Time")

        plt.ylabel("Predicted AQI")

        plt.title("Prediction Uncertainty Estimation")

        plt.legend()

        save_path = os.path.join(
            OUTPUT_DIR,
            "prediction_uncertainty.png"
        )

        plt.savefig(save_path)

        plt.show()

        print("Saved uncertainty plot:", save_path)


# --------------------------------
# Run Module
# --------------------------------

def run_self_evaluation():

    engine = UncertaintyEstimator()

    X = engine.build_sequences()

    print("Running Monte Carlo Dropout...")

    mean_pred, std_pred = engine.mc_dropout_prediction(X)

    engine.plot_uncertainty(mean_pred, std_pred)


if __name__ == "__main__":

    run_self_evaluation()