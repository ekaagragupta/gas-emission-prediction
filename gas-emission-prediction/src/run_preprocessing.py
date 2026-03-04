import pandas as pd

from preprocessing import EmissionPreprocessor
from feature_engineering import (
    create_time_features,
    create_lag_features
)


raw_data = pd.read_csv("data/raw/delhi_emissions.csv")  # change filename if needed


preprocessor = EmissionPreprocessor()

clean_df = preprocessor.preprocess(raw_data)

# Feature engineering
clean_df = create_time_features(clean_df)
clean_df = create_lag_features(clean_df)

clean_df = clean_df.drop(columns=["Date"])
clean_df = clean_df.dropna()


clean_df.to_csv("data/processed/emissions_clean.csv", index=False)

print("Processed dataset saved.")
print("Shape:", clean_df.shape)