import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "emissions_clean.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "causal")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class CausalDiscoveryEngine:

    def __init__(self):

        print("Loading dataset...")

        self.df = pd.read_csv(DATA_PATH)

        if "Date" in self.df.columns:
            self.df = self.df.drop(columns=["Date"])

        print("Dataset shape:", self.df.shape)


    # -----------------------------------
    # Granger Causality Test
    # -----------------------------------

    def run_granger_test(self, target="AQI", max_lag=5):

        print("Running Granger Causality Tests...")

        results = []

        for col in self.df.columns:

            if col == target:
                continue

            test_df = self.df[[target, col]].dropna()

            try:

                test_result = grangercausalitytests(
                    test_df,
                    maxlag=max_lag,
                    verbose=False
                )

                p_values = [
                    test_result[i + 1][0]['ssr_ftest'][1]
                    for i in range(max_lag)
                ]

                min_p = np.min(p_values)

                results.append({
                    "feature": col,
                    "p_value": min_p
                })

            except:

                continue

        df_results = pd.DataFrame(results)

        df_results = df_results.sort_values("p_value")

        return df_results


    # -----------------------------------
    # Plot causal strength
    # -----------------------------------

    def plot_causal_strength(self, df_results):

        top = df_results.head(10)

        plt.figure(figsize=(8,5))

        plt.barh(
            top["feature"],
            -np.log10(top["p_value"])
        )

        plt.xlabel("-log10(p-value)")
        plt.title("Causal Influence on AQI")

        plt.gca().invert_yaxis()

        save_path = os.path.join(
            OUTPUT_DIR,
            "causal_influence.png"
        )

        plt.savefig(save_path)

        plt.show()

        print("Saved causal plot:", save_path)


# -----------------------------------
# Entry Point
# -----------------------------------

def run_causal_discovery():

    engine = CausalDiscoveryEngine()

    results = engine.run_granger_test()

    print("\nTop causal variables affecting AQI:\n")

    print(results.head(10))

    engine.plot_causal_strength(results)


if __name__ == "__main__":

    run_causal_discovery()