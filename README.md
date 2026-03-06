
# GETHER

###  Emission Temporal Hybrid Explainable Regressor

A human-centered AI system for **air pollution understanding, forecasting, and policy simulation**.
GETHER combines **temporal deep learning, causal discovery, counterfactual reasoning, and explainable AI** to help researchers, policymakers, and the public understand how emissions affect **AQI (Air Quality Index)** and what interventions could improve it.

---

# Overview

Air pollution prediction alone is not enough.
GETHER focuses on **understanding pollution**, not just predicting it.

The system answers questions like:

* What pollutants cause AQI spikes?
* What happens if emissions are reduced?
* Which environmental factors matter most?
* How reliable are model predictions?

The platform integrates **data infrastructure, machine learning, explainability, and policy simulation** into a single research pipeline.

---

# Core Capabilities

**1. Temporal AQI Prediction**

Predict AQI using a hybrid system combining:

* Feature engineering
* Sequence modeling
* LSTM forecasting

---

**2. Causal Discovery**

Identify **true causal relationships** between pollutants and AQI using:

* Granger causality
* Dynamic causal graphs
* Spatiotemporal correlation analysis

---

**3. Counterfactual Simulation**

Simulate environmental policies such as:

* emission reductions
* traffic control
* industrial regulation

Example:

> If PM2.5 emissions decrease by 20%, what happens to AQI over the next 30 days?

---

**4. Explainable AI**

Model explanations generated using:

* SHAP feature importance
* Local prediction explanation
* Feature contribution ranking

---

**5. Self-Evaluation**

The system evaluates its own predictions using:

* prediction confidence
* uncertainty estimation
* model reliability metrics

---

**6. Interactive Dashboard**

A Streamlit dashboard enables:

* real-time AQI prediction
* causal graph exploration
* counterfactual policy simulation
* downloadable analysis reports

---

# Prediction Example

The model predicts AQI trends based on pollutant history.

![AQI Prediction Graph](gas-emission-prediction/plots/aqi_predictions.png)

The graph compares:

* **Actual AQI**
* **Predicted AQI**

This helps evaluate model performance and temporal patterns.

---

# System Architecture

![System Architecture](gas-emission-prediction/plots/Diagram.png)

The architecture integrates:

* Data acquisition
* Preprocessing pipeline
* Temporal ML models
* Explainable AI
* Policy simulation engine
* Dashboard interface

---

# Pipeline Phases

## Phase 1 — Data Infrastructure & Validation

* Satellite data acquisition
* Weather API integration
* Raw data storage
* Preprocessing pipeline
* Feature engineering
* Train / validation / test split
* Data validation reports

---

## Phase 2 — Baseline Modeling

Baseline models provide initial performance benchmarks.

Models used:

* Linear Regression
* Random Forest
* Basic LSTM

Outputs include:

* model comparison
* error analysis
* feature importance

---

## Phase 3 — Advanced LSTM Development

The final forecasting system uses a deep temporal architecture.

Architecture:

```
Input Sequence
   ↓
LSTM (128)
   ↓
Dropout
   ↓
LSTM (64)
   ↓
Dense (32)
   ↓
AQI Prediction
```

Enhancements include:

* rolling statistical features
* lag features
* Bayesian hyperparameter tuning
* model averaging

---

## Phase 4 — Causal Discovery Engine

Goal: identify **cause-effect relationships** in emissions data.

Techniques used:

* Granger causality testing
* spatial correlation analysis
* dynamic causal graph construction

Output:

```
PM2.5 → AQI
NO2 → AQI
SO2 → AQI
```

---

## Phase 5 — Explainable AI & Uncertainty

Explainability is provided through:

**SHAP Analysis**

* feature importance
* local explanations
* model transparency

**Uncertainty Estimation**

* confidence intervals
* prediction stability
* model reliability scoring

Automated reports are generated for researchers and policy analysts.

---

## Phase 6 — Counterfactual Policy Simulator

A simulation engine allows testing environmental interventions.

Example simulation:

```
Policy: reduce NO2 emissions by 25%

Baseline AQI forecast: 210
Counterfactual AQI forecast: 168

Average improvement: 20%
```

Policies can be ranked based on:

* emission reduction
* AQI improvement
* environmental impact

---

## Phase 7 — Dashboard Deployment

The final application provides:

* interactive AQI prediction
* causal graph visualization
* policy simulation interface
* explainability reports

Built with:

```
Streamlit
TensorFlow
SHAP
Scikit-learn
```

---

# Project Structure

```
gas-emission-prediction
│
├── data
│   ├── raw
│   ├── processed
│
├── notebooks
│
├── src
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── run_preprocessing.py
│   ├── train_lstm.py
│   ├── visualize_predictions.py
│
│   ├── models
│   │   ├── baseline_models.py
│   │   └── advanced_lstm.py
│
│   ├── causal_discovery.py
│   ├── counterfactual_analysis.py
│   ├── explainability.py
│   └── self_evaluation.py
│
├── models
│   └── aqi_lstm_model.h5
│
└── README.md
```

---

# Model Performance

Baseline experiments:

| Model             | R²    | MAE   |
| ----------------- | ----- | ----- |
| Linear Regression | 0.13  | 0.066 |
| Random Forest     | -0.64 | 0.102 |
| Basic LSTM        | -0.37 | 0.092 |

Advanced LSTM:

```
Test Loss : 0.015
Test MAE  : 0.094
```

---

# Installation

Clone repository:

```
git clone https://github.com/yourusername/gas-emission-prediction.git
cd gas-emission-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Run Pipeline

Preprocess data

```
python src/run_preprocessing.py
```

Train models

```
python src/train_lstm.py
```

Visualize predictions

```
python src/visualize_predictions.py
```

---

# Example Use Cases

**Researchers**

* pollution trend analysis
* environmental modeling
* causal inference studies

**Policy Makers**

* policy intervention evaluation
* emission reduction planning
* air quality forecasting

**Public**

* AQI awareness
* pollution insights
* environmental impact visualization

---

# Future Extensions

Planned improvements include:

* Graph Neural Networks for spatial pollution modeling
* transformer-based temporal forecasting
* real-time satellite data ingestion
* multi-city causal networks


---

# Author

Ekaagra Gupta -
B.Tech AI & ML

---
