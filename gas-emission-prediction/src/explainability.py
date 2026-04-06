"""
GETHER — Generative AI-driven Emission Temporal Hybrid Explainable Regressor
File: src/explainability.py

Explainability Module using SHAP.
Computes global feature importance and local explanations
for AQI predictions without modifying any existing modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths (relative to project root)
# ─────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "aqi_lstm_model.h5")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
DATA_PATH    = os.path.join("data", "processed", "emissions_clean.csv")
OUTPUT_DIR   = os.path.join("outputs", "explainability")


# ─────────────────────────────────────────────
# Helper: load artefacts
# ─────────────────────────────────────────────
def _load_model():
    """Load the trained Keras LSTM model."""
    from tensorflow import keras  # lazy import keeps module light
    model = keras.models.load_model(MODEL_PATH)
    return model


def _load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def _load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


# ─────────────────────────────────────────────
# Sequence builder (mirrors train_lstm.py logic)
# ─────────────────────────────────────────────
def _build_sequences(df: pd.DataFrame, seq_len: int = 30):
    """
    Re-create LSTM input sequences from the processed dataframe.
    Target column: AQI.  Feature columns: everything else (non-Date).
    Returns X (3-D), y (1-D), feature_names (list).
    """
    feature_cols = [c for c in df.columns if c not in ("Date", "AQI")]
    target_col   = "AQI"

    data_X = df[feature_cols].values
    data_y = df[target_col].values

    X, y = [], []
    for i in range(seq_len, len(data_X)):
        X.append(data_X[i - seq_len: i])   # (seq_len, n_features)
        y.append(data_y[i])

    return np.array(X), np.array(y), feature_cols


# ─────────────────────────────────────────────
# Core: SHAP wrapper around LSTM
# ─────────────────────────────────────────────
class AQIExplainer:
    """
    Wraps the trained LSTM model and exposes SHAP-based explanations.

    Attributes
    ----------
    model        : Keras model
    scaler       : fitted MinMaxScaler
    feature_names: list[str]
    shap_values  : np.ndarray  (set after calling compute_shap_values)
    X_flat       : np.ndarray  2-D flattened sequences (samples × features_flat)
    """

    def __init__(self, seq_len: int = 30, n_background: int = 100):
        """
        Parameters
        ----------
        seq_len      : sequence length used during training (default 30)
        n_background : number of background samples for SHAP KernelExplainer
        """
        self.seq_len      = seq_len
        self.n_background = n_background

        self.model         = _load_model()
        self.scaler        = _load_scaler()
        self.df            = _load_data()

        self.X, self.y, self.feature_names = _build_sequences(self.df, seq_len)

        # Flatten sequences → (samples, seq_len * n_features) for SHAP
        n_samples, sl, n_feat = self.X.shape
        self.X_flat = self.X.reshape(n_samples, sl * n_feat)

        # Human-readable flat feature names: "feat@t-29", "feat@t-28", …
        self.flat_feature_names = [
            f"{fn}@t-{seq_len - 1 - t}"
            for t in range(seq_len)
            for fn in self.feature_names
        ]

        # Aggregated feature names (averaged over time steps) for summary plots
        self.agg_feature_names = self.feature_names

        self.shap_values     = None
        self.shap_values_agg = None   # (samples, n_features) — mean |SHAP| over time
        self._explainer      = None

    # ── Prediction wrapper (required by SHAP) ──────────────────────────────
    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        """Reshape flat array back to 3-D and run inference."""
        n = X_flat.shape[0]
        X3d = X_flat.reshape(n, self.seq_len, len(self.feature_names))
        preds = self.model.predict(X3d, verbose=0)
        return preds.flatten()

    # ── SHAP computation ───────────────────────────────────────────────────
    def compute_shap_values(self, n_explain: int = 200):
        """
        Compute SHAP values using KernelExplainer.

        Parameters
        ----------
        n_explain : number of test samples to explain (keep small for speed)
        """
        print("[SHAP] Selecting background and explanation samples …")
        bg_idx  = np.random.choice(len(self.X_flat), self.n_background, replace=False)
        exp_idx = np.random.choice(len(self.X_flat), n_explain,         replace=False)

        background  = self.X_flat[bg_idx]
        X_to_explain = self.X_flat[exp_idx]

        print(f"[SHAP] Building KernelExplainer (background={self.n_background}) …")
        self._explainer  = shap.KernelExplainer(self._predict_flat, background)

        print(f"[SHAP] Computing SHAP values for {n_explain} samples …")
        self.shap_values = self._explainer.shap_values(X_to_explain, nsamples=200)
        self.X_explain   = X_to_explain

        # Aggregate: reshape to (n_explain, seq_len, n_feat) then mean over time
        sv_3d = self.shap_values.reshape(n_explain, self.seq_len, len(self.feature_names))
        self.shap_values_agg = sv_3d.mean(axis=1)          # (n_explain, n_feat)
        self.X_explain_agg   = X_to_explain.reshape(
            n_explain, self.seq_len, len(self.feature_names)
        ).mean(axis=1)                                      # (n_explain, n_feat)

        print("[SHAP] Done.")
        return self.shap_values

    # ── Global importance ──────────────────────────────────────────────────
    def global_feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of mean absolute SHAP value per feature,
        aggregated over the sequence (time-averaged).
        """
        if self.shap_values_agg is None:
            raise RuntimeError("Call compute_shap_values() first.")

        importance = np.abs(self.shap_values_agg).mean(axis=0)
        df_imp = pd.DataFrame({
            "feature":    self.agg_feature_names,
            "mean_|SHAP|": importance
        }).sort_values("mean_|SHAP|", ascending=False).reset_index(drop=True)
        return df_imp

    # ── Local explanation ──────────────────────────────────────────────────
    def local_explanation(self, sample_idx: int = 0) -> pd.DataFrame:
        """
        Return SHAP values for a single explained sample (time-averaged).

        Parameters
        ----------
        sample_idx : index into the explained subset (0 … n_explain-1)
        """
        if self.shap_values_agg is None:
            raise RuntimeError("Call compute_shap_values() first.")

        sv  = self.shap_values_agg[sample_idx]
        val = self.X_explain_agg[sample_idx]

        df_local = pd.DataFrame({
            "feature":    self.agg_feature_names,
            "shap_value": sv,
            "feature_value": val
        }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)
        return df_local

    # ─────────────────────────────────────────────────────────────────────
    # Visualisations
    # ─────────────────────────────────────────────────────────────────────
    def _ensure_output_dir(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def plot_global_importance(self, top_n: int = 15, save: bool = True):
        """Bar chart of top-N most important features (global)."""
        df_imp = self.global_feature_importance().head(top_n)

        self._ensure_output_dir()
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_imp["feature"][::-1], df_imp["mean_|SHAP|"][::-1],
                       color="#2196F3", edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"GETHER — Global Feature Importance (top {top_n})", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save:
            path = os.path.join(OUTPUT_DIR, "global_importance.png")
            fig.savefig(path, dpi=150)
            print(f"[Plot] Saved → {path}")
        plt.show()
        return fig

    def plot_shap_summary(self, save: bool = True):
        """SHAP beeswarm / summary plot (aggregated over time steps)."""
        if self.shap_values_agg is None:
            raise RuntimeError("Call compute_shap_values() first.")

        self._ensure_output_dir()
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values_agg,
            self.X_explain_agg,
            feature_names=self.agg_feature_names,
            show=False,
            plot_type="dot"
        )
        plt.title("GETHER — SHAP Summary Plot", fontsize=13)
        plt.tight_layout()

        if save:
            path = os.path.join(OUTPUT_DIR, "shap_summary.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved → {path}")
        plt.show()

    def plot_local_explanation(self, sample_idx: int = 0,
                               top_n: int = 10, save: bool = True):
        """
        Waterfall-style bar chart for a single prediction.
        Positive SHAP → increases AQI; negative → decreases AQI.
        """
        df_local = self.local_explanation(sample_idx).head(top_n)
        self._ensure_output_dir()

        colors = ["#E53935" if v > 0 else "#1E88E5" for v in df_local["shap_value"]]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_local["feature"][::-1], df_local["shap_value"][::-1],
                       color=colors[::-1], edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value  (impact on AQI prediction)", fontsize=11)
        ax.set_title(
            f"GETHER — Local Explanation  (sample #{sample_idx})", fontsize=13
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save:
            path = os.path.join(OUTPUT_DIR, f"local_explanation_sample{sample_idx}.png")
            fig.savefig(path, dpi=150)
            print(f"[Plot] Saved → {path}")
        plt.show()
        return fig

    def plot_temporal_shap(self, feature: str, sample_idx: int = 0,
                           save: bool = True):
        """
        Show how SHAP values for a single feature evolve over the 30 time steps
        for one prediction instance.
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        n_feat  = len(self.feature_names)
        feat_idx = self.feature_names.index(feature)

        # Reshape: (n_explain, seq_len, n_features)
        sv_3d = self.shap_values.reshape(-1, self.seq_len, n_feat)
        sv_feat = sv_3d[sample_idx, :, feat_idx]   # (seq_len,)

        self._ensure_output_dir()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(self.seq_len), sv_feat, marker="o", color="#7B1FA2", linewidth=1.5)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time step (t-29  →  t)", fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.set_title(
            f"GETHER — Temporal SHAP: {feature}  (sample #{sample_idx})", fontsize=13
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save:
            path = os.path.join(OUTPUT_DIR, f"temporal_shap_{feature}_s{sample_idx}.png")
            fig.savefig(path, dpi=150)
            print(f"[Plot] Saved → {path}")
        plt.show()
        return fig

def run_explainability(
    seq_len:      int  = 30,
    n_background: int  = 100,
    n_explain:    int  = 200,
    top_n:        int  = 15,
    local_sample: int  = 0,
    temporal_feature: str = "PM2.5",
    save_plots:   bool = True
) -> AQIExplainer:
    """
    Full explainability pipeline:

    1. Load model, scaler, and processed data.
    2. Build LSTM sequences.
    3. Compute SHAP values via KernelExplainer.
    4. Print global importance table.
    5. Generate and optionally save:
       - Global importance bar chart
       - SHAP summary (beeswarm) plot
       - Local waterfall chart for one sample
       - Temporal SHAP trace for one feature

    Returns the fitted AQIExplainer for further use.
    """
    print("=" * 60)
    print("  GETHER — Explainability Module")
    print("=" * 60)

    explainer = AQIExplainer(seq_len=seq_len, n_background=n_background)
    print(f"[Info] Features : {len(explainer.feature_names)}")
    print(f"[Info] Sequences: {len(explainer.X)}")

    explainer.compute_shap_values(n_explain=n_explain)

    # ── Global importance ──────────────────────────────────────────────────
    print("\n── Global Feature Importance ──────────────────────────────")
    df_imp = explainer.global_feature_importance()
    print(df_imp.head(top_n).to_string(index=False))

    explainer.plot_global_importance(top_n=top_n, save=save_plots)

    # ── Summary plot ───────────────────────────────────────────────────────
    explainer.plot_shap_summary(save=save_plots)

    # ── explanation for one sample ───────────────────────────────────────────────
    print(f"\n── Local Explanation  (sample #{local_sample}) ──────────────")
    df_local = explainer.local_explanation(local_sample)
    print(df_local.head(10).to_string(index=False))

    explainer.plot_local_explanation(sample_idx=local_sample,
                                     top_n=top_n, save=save_plots)

    # ── Temporal SHAP ──────────────────────────────────────────────────────
    if temporal_feature in explainer.feature_names:
        explainer.plot_temporal_shap(feature=temporal_feature,
                                     sample_idx=local_sample,
                                     save=save_plots)
    else:
        print(f"[Warning] '{temporal_feature}' not in feature list; "
              f"skipping temporal SHAP plot.")

    print("\n[Done] Explainability module completed successfully.")
    return explainer

if __name__ == "__main__":
    run_explainability(
        seq_len=30,
        n_background=100,
        n_explain=200,
        top_n=15,
        local_sample=0,
        temporal_feature="PM2.5",
        save_plots=True
    )