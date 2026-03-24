"""
src/model.py
Trains and saves the ML model for inverse mapping of heat treatment parameters.
Features: Composition + Target Hardness
Targets: Austenitizing Temp, Tempering Temp, Tempering Time
"""
import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from .data_processing import load_processed_data
except ImportError:
    from data_processing import load_processed_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "heat_treatment_model.pkl")

def get_inverse_split(df: pd.DataFrame):
    """
    Splits the dataframe for the Inverse Mapping model.
    Features: Composition (incl. P, S) + Hardness
    Targets (ordered for RegressorChain): Austenitizing_Temp_C → Tempering_Temp_C → Tempering_Time_Hours
    """
    feature_cols = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 'P', 'S', 'Hardness_HRC']
    target_cols  = ['Austenitizing_Temp_C', 'Tempering_Temp_C', 'Tempering_Time_Hours']
    X = df[feature_cols]
    y = df[target_cols]
    return X, y

def train_model():
    """Trains the RegressorChain XGBoost model with optional Optuna tuning and evaluates it."""
    print("Loading data...")
    df = load_processed_data()
    X, y = get_inverse_split(df)
    print(f"Features: {list(X.columns)}")
    print(f"Targets: {list(y.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- B2: Optuna hyperparameter tuning ---
    if HAS_OPTUNA:
        print("Running Optuna hyperparameter search (30 trials)...")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }
            m = RegressorChain(XGBRegressor(**params))
            scores = cross_val_score(m, X_train, y_train, cv=3, scoring='r2')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        best_params['random_state'] = 42
        print(f"Best params: {best_params}")
    else:
        print("Optuna not available — using default params. Run: pip install optuna")
        best_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}

    # Save best params
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH.replace('.pkl', '_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    # --- B1: Train with RegressorChain (ordered targets improve correlation capture) ---
    print("Training RegressorChain XGBoost ensemble (5 seeds)...")
    ensemble = []
    for seed in range(5):
        params = dict(best_params, random_state=seed)
        model = RegressorChain(XGBRegressor(**params))
        model.fit(X_train, y_train)
        ensemble.append(model)

    # --- Evaluate on test set using mean of ensemble ---
    print("Evaluating ensemble...")
    all_preds = np.stack([m.predict(X_test) for m in ensemble])  # (5, N, 3)
    y_pred_mean = all_preds.mean(axis=0)
    y_pred_std  = all_preds.std(axis=0)

    for i, col in enumerate(y.columns):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred_mean[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred_mean[:, i])
        avg_std = y_pred_std[:, i].mean()
        print(f"--- {col} ---")
        print(f"  MAE: {mae:.2f}  |  R²: {r2:.2f}  |  Avg±std: {avg_std:.2f}")

    # Save all 5 models
    for i, m in enumerate(ensemble):
        path = MODEL_PATH.replace('.pkl', f'_ensemble_{i}.pkl')
        joblib.dump(m, path)
    print(f"\n5-model ensemble saved to {os.path.dirname(MODEL_PATH)}/")

if __name__ == "__main__":
    train_model()
