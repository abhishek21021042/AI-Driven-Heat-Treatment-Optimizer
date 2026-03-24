"""
src/optimizer.py
Inference engine for the Heat Treatment Optimizer.
Uses a 5-model ensemble (RegressorChain) to predict parameters with uncertainty bands.
"""
import os
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_BASE = os.path.join(MODEL_DIR, "heat_treatment_model")

_ensemble = None

def load_ensemble():
    global _ensemble
    if _ensemble is None:
        ensemble = []
        for i in range(5):
            path = f"{MODEL_BASE}_ensemble_{i}.pkl"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ensemble model {i} not found at {path}. Run python src/model.py first.")
            ensemble.append(joblib.load(path))
        _ensemble = ensemble
    return _ensemble

# Feature columns must match model.py training
FEATURE_COLS = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 'P', 'S', 'Hardness_HRC']

def optimize_heat_treatment(comp_dict: dict, target_hardness: float) -> dict:
    """
    Predicts optimal heat treatment parameters to achieve the target hardness.

    Args:
        comp_dict: Dict with keys ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo', 'V', 'P', 'S']
        target_hardness: Desired final hardness in HRC

    Returns:
        dict with mean predictions, std (uncertainty), quenchant, Ceq, DI_mm
    """
    ensemble = load_ensemble()

    input_data = {col: comp_dict.get(col.replace('Hardness_HRC', ''), 0.0) for col in FEATURE_COLS}
    input_data['Hardness_HRC'] = target_hardness
    df_input = pd.DataFrame([input_data])[FEATURE_COLS]

    all_preds = np.stack([m.predict(df_input)[0] for m in ensemble])  # (5, 3)
    means = all_preds.mean(axis=0)
    stds  = all_preds.std(axis=0)

    # --- C3: Ceq (Dearden & O'Neill) ---
    C   = comp_dict.get('C',  0.0)
    Mn  = comp_dict.get('Mn', 0.0)
    Cr  = comp_dict.get('Cr', 0.0)
    Mo  = comp_dict.get('Mo', 0.0)
    V   = comp_dict.get('V',  0.0)
    Ni  = comp_dict.get('Ni', 0.0)
    ceq = C + Mn/6 + (Cr + Mo + V)/5 + Ni/15

    # --- C3: Ideal Critical Diameter DI (Grossmann) ---
    # DI = f(C) * multipliers for each element
    # Carbon base hardenability from Bain-Paxton factor approximation
    di_base = 0.54 * np.sqrt(C) if C > 0 else 0.1
    f_mn  = 1 + 4.37  * Mn
    f_cr  = 1 + 2.16  * Cr
    f_mo  = 1 + 3.0   * Mo
    f_ni  = 1 + 0.363 * Ni
    f_si  = 1 + 1.5   * comp_dict.get('Si', 0.0)
    di_mm = round(di_base * f_mn * f_cr * f_mo * f_ni * f_si * 25.4, 1)  # inches → mm

    # --- Quenchant logic ---
    is_air_hardening = (Cr + Mo) >= 4.5
    if target_hardness <= 25.0:
        quench_medium = "Sand Cooling 🏖️"
    elif is_air_hardening:
        quench_medium = "Air Quench 💨"
    elif C >= 0.45 or ceq > 0.65:
        quench_medium = "Oil Quench 🛢️"
    else:
        quench_medium = "Water Quench 💧"

    return {
        'Austenitizing_Temp_C':      float(np.round(means[0], 1)),
        'Austenitizing_Temp_std':     float(np.round(stds[0],  1)),
        'Tempering_Temp_C':           float(np.round(means[1], 1)),
        'Tempering_Temp_std':         float(np.round(stds[1],  1)),
        'Tempering_Time_Hours':       float(np.round(means[2], 2)),
        'Tempering_Time_std':         float(np.round(stds[2],  2)),
        'Quench_Medium':              quench_medium,
        'Ceq':                        float(np.round(ceq, 3)),
        'DI_mm':                      float(di_mm),
    }

def predict_tempering_curve(comp_dict: dict, hardness_range=None) -> pd.DataFrame:
    """
    Predict Tempering Temperature across a range of desired hardness values.
    Returns a DataFrame of {'Hardness_HRC', 'Tempering_Temp_C'}.
    """
    if hardness_range is None:
        hardness_range = np.arange(20, 66, 2.0)

    records = []
    for hrc in hardness_range:
        try:
            res = optimize_heat_treatment(comp_dict, hrc)
            records.append({'Hardness_HRC': hrc, 'Tempering_Temp_C': res['Tempering_Temp_C']})
        except Exception:
            pass
    return pd.DataFrame(records)

if __name__ == "__main__":
    test_comp = {'C': 0.4, 'Mn': 0.8, 'Si': 0.2, 'Cr': 1.0, 'Mo': 0.2}
    res = optimize_heat_treatment(test_comp, 45.0)
    print(res)
