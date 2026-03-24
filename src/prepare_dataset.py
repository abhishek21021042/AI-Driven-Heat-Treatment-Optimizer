"""
prepare_dataset.py
Reads raw CSV data, cleans, calculates missing features, optionally augments,
and saves the final dataset to data/processed/.
"""

import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw/Tempering data for carbon and low alloy steels - Raiipa(csv).csv"
PROCESSED_DATA_PATH = "data/processed/heat_treatment_dataset.csv"

def calculate_ac3(row):
    """Calculate Ac3 temperature based on Andrews formula."""
    # Ac3 = 910 - 203*C - 15.2*Ni + 44.7*Si + 104*V + 31.5*Mo
    c = row.get('C', 0.0)
    ni = row.get('Ni', 0.0)
    si = row.get('Si', 0.0)
    v = row.get('V', 0.0)
    mo = row.get('Mo', 0.0)
    
    # Fill nan with 0 for composition
    c = 0.0 if pd.isna(c) else c
    ni = 0.0 if pd.isna(ni) else ni
    si = 0.0 if pd.isna(si) else si
    v = 0.0 if pd.isna(v) else v
    mo = 0.0 if pd.isna(mo) else mo
    
    ac3 = 910 - 203 * c - 15.2 * ni + 44.7 * si + 104 * v + 31.5 * mo
    return ac3

def augment_data(df, target_size=5000):
    """Augment dataset by adding small Gaussian noise to continuous variables."""
    current_size = len(df)
    if current_size >= target_size:
        return df
        
    num_to_generate = target_size - current_size
    
    # Sample randomly from existing rows
    synthetic_df = df.sample(n=num_to_generate, replace=True).copy()
    
    # Add small noise
    # Composition noise (±0.02%)
    comp_cols = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 'P', 'S']
    for col in comp_cols:
        noise = np.random.normal(0, 0.01, size=len(synthetic_df))
        synthetic_df[col] = np.clip(synthetic_df[col] + noise, 0, None) # Ensure >= 0
        
    # Process variables noise
    synthetic_df['Tempering_Temp_C'] += np.random.normal(0, 5, size=len(synthetic_df)) # ± ~15°C
    synthetic_df['Tempering_Time_Hours'] += np.random.normal(0, 0.1, size=len(synthetic_df)) # ± ~0.3 hours
    synthetic_df['Tempering_Time_Hours'] = np.clip(synthetic_df['Tempering_Time_Hours'], 0.1, None)
    
    # Small noise to hardness
    synthetic_df['Hardness_HRC'] += np.random.normal(0, 1.0, size=len(synthetic_df))
    
    # Re-calculate Austenitizing Temp for the synthetic data to maintain physical physics
    synthetic_df['Austenitizing_Temp_C'] = synthetic_df.apply(calculate_ac3, axis=1) + 50 + np.random.normal(0, 5, size=len(synthetic_df))
    
    return pd.concat([df, synthetic_df], ignore_index=True)

def main():
    np.random.seed(42)  # A1: reproducibility
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Rename columns based on Raiipa dataset
    col_mapping = {
        'C (%wt)': 'C',
        'Mn (%wt)': 'Mn',
        'Si (%wt)': 'Si',
        'Ni (%wt)': 'Ni',
        'Cr (%wt)': 'Cr',
        'Mo (%wt)': 'Mo',
        'V (%wt)': 'V',
        'P (%wt)': 'P',  # A2: Phosphorus
        'S (%wt)': 'S',  # A2: Sulfur
        'Tempering temperature (ºC)': 'Tempering_Temp_C',
        'Tempering time (s)': 'Tempering_Time_s',
        'Final hardness (HRC) - post tempering': 'Hardness_HRC'
    }
    
    # Select only available columns from mapping
    available_cols = [c for c in col_mapping.keys() if c in df.columns]
    df = df[available_cols].rename(columns=col_mapping)
    
    # Drop rows with missing target or crucial variables
    df = df.dropna(subset=['Hardness_HRC', 'Tempering_Temp_C', 'Tempering_Time_s', 'C'])
    
    # Convert '?' to NaN for HRC if it became string
    df['Hardness_HRC'] = pd.to_numeric(df['Hardness_HRC'], errors='coerce')
    df = df.dropna(subset=['Hardness_HRC'])
    
    # Convert Tempering_Time_s to Hours
    df['Tempering_Time_s'] = pd.to_numeric(df['Tempering_Time_s'], errors='coerce')
    df['Tempering_Time_Hours'] = df['Tempering_Time_s'] / 3600.0
    df = df.drop(columns=['Tempering_Time_s'])
    
    # INDUSTRIAL OPTIMIZATION:
    # The raw dataset contains lab experiments with tempering times up to 32 hours!
    # For a time/energy-efficient web tool, we should only teach the model cycles that take <= 4 hours.
    df = df[df['Tempering_Time_Hours'] <= 4.0]
    
    # Fill remaining missing composition with 0
    comp_cols = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 'P', 'S']
    for col in comp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
            
    print(f"Cleaned dataset shape: {df.shape}")
    
    # Calculate expected Austenitizing Temperature (Ac3 + 50°C + minor random noise)
    # This allows the ML model to learn that variations around this mean are typical
    df['Austenitizing_Temp_C'] = df.apply(calculate_ac3, axis=1) + 50 + np.random.normal(0, 5, size=len(df))
    
    # Augment data to ~5000 rows to ensure robust ML training
    print("Augmenting data to ~5000 rows...")
    df_augmented = augment_data(df, target_size=5000)
    
    # Final column ordering
    feature_cols = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 'P', 'S',
                    'Austenitizing_Temp_C', 'Tempering_Temp_C', 'Tempering_Time_Hours']
    target_col = ['Hardness_HRC']
    final_df = df_augmented[feature_cols + target_col]
    
    print(f"Final augmented dataset shape: {final_df.shape}")
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
