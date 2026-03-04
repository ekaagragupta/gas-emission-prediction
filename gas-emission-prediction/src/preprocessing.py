import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import List


class EmissionPreprocessor:
    """
    Complete preprocessing pipeline.
    """

    def __init__(self, scaler_path: str = "models/scaler.pkl"):
        self.scaler = MinMaxScaler()
        self.scaler_path = scaler_path

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date")
        df = df.drop(columns=["City", "AQI_Bucket"])
        df = df.ffill(limit=6)
        df = df.interpolate(method="linear")
        return df

    def remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3 * std) &
                    (df[col] <= mean + 3 * std)]
        return df

    def normalize(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df[columns] = self.scaler.fit_transform(df[columns])

        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df["Date"] = pd.to_datetime(df["Date"])
        df = self.remove_duplicates(df)
        df = self.handle_missing(df)

        numeric_cols = df.select_dtypes(include=np.number).columns
        df = self.remove_outliers(df, numeric_cols)

        df = self.normalize(df, numeric_cols)

        return df