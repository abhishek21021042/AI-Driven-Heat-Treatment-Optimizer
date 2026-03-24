"""
src/data_processing.py
Standardized interface to load the processed dataset and provide feature/target splits.
"""
import pandas as pd
import numpy as np
import os

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "heat_treatment_dataset.csv")

def load_processed_data() -> pd.DataFrame:
    """Loads the fully cleaned and processed heat treatment dataset."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATA_PATH}. Please run src/prepare_dataset.py first.")
    return pd.read_csv(PROCESSED_DATA_PATH)

def get_feature_target_split(df: pd.DataFrame):
    """
    Splits the dataframe into X (features) and y (target).
    
    Features: Composition (C, Mn, Si, Ni, Cr, Mo, V) + Austenitizing_Temp_C + Tempering_Temp_C + Tempering_Time_Hours
    Target: Hardness_HRC
    """
    feature_cols = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 
                    'Austenitizing_Temp_C', 'Tempering_Temp_C', 'Tempering_Time_Hours']
    target_col = 'Hardness_HRC'
    
    X = df[feature_cols]
    y = df[target_col]
    return X, y
