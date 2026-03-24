# 🔥 AI-Driven Heat Treatment Optimizer

**🔴  Demo link :** [Click here to launch the web application!](https://ai-driven-heat-treatment-optimizergit-efjuwo6tuo7m2a8zmvjviw.streamlit.app/)

An **open-source machine learning web application** that predicts the optimal, time- and energy-efficient heat treatment cycle for low carbon and low-alloy steels. Instead of handbook charts or expensive trial-and-error, enter your alloy composition and desired hardness — the AI ensemble instantly recommends the full process window.

> **⚠️ Scope Notice:** This model is trained exclusively on **low carbon and low-alloy steel** heat treatment data. Predictions for high-alloy steels, tool steels, or high-carbon steels (>0.6% C) will be extrapolative.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Grade Presets** | Quick-select from common AISI grades (1020, 1040, 4140, 4340) to auto-fill composition |
| **4 Core Predictions** | Austenitizing Temp, Tempering Temp, Tempering Time, Quench Medium |
| **Uncertainty Bands** | ± std across ensemble for every continuous output |
| **Hardenability** | Real-time Carbon Equivalent (Ceq) and Ideal Critical Diameter (DI) from Grossmann formula |
| **Tempering Curve** | Interactive Plotly chart of predicted hardness vs. tempering temperature |
| **Hardness Constraint** | Burns-Moore-Archer maximum HRC limit enforced dynamically per carbon level |
| **Single-screen UI** | No scrolling needed — compact dark dashboard with custom CSS |

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/abhishek21021042/AI-Driven-Heat-Treatment-Optimizer.git
cd AI-Driven-Heat-Treatment-Optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app
python -m streamlit run src/app.py
```

The app opens at **http://localhost:8501**.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit >= 1.30` | Web UI framework |
| `xgboost` | Main ML algorithm |
| `scikit-learn` | `RegressorChain`, metrics, train/test split |
| `optuna` | Bayesian hyperparameter optimization |
| `pandas` / `numpy` | Data processing and numerical operations |
| `plotly` | Interactive tempering curve chart |
| `joblib` | Model serialization |
| `pytest` | Unit testing |

---

## 📊 Raw Data Details

**Source Dataset:** `Tempering data for carbon and low alloy steels - Raiipa.csv`

This is a published academic dataset of real experimental heat treatment records for carbon and low-alloy steels. Each row represents an actual laboratory tempering experiment.

**Original Raw Columns:**
| Column | Unit | Description |
|---|---|---|
| `C (%wt)` | % | Carbon content |
| `Mn (%wt)` | % | Manganese content |
| `Si (%wt)` | % | Silicon content |
| `Ni (%wt)` | % | Nickel content |
| `Cr (%wt)` | % | Chromium content |
| `Mo (%wt)` | % | Molybdenum content |
| `V (%wt)` | % | Vanadium content |
| `P (%wt)` | % | Phosphorus content |
| `S (%wt)` | % | Sulfur content |
| `Tempering temperature (ºC)` | °C | Temperature of tempering stage |
| `Tempering time (s)` | seconds | Duration of tempering step |
| `Final hardness (HRC) - post tempering` | HRC | Resulting hardness after process |

---

## 🔧 Data Processing Pipeline (`prepare_dataset.py`)

The raw CSV is transformed through the following pipeline before ML training:

1. **Column Mapping & Renaming** — Raw column names are standardized (e.g. `Tempering time (s)` → `Tempering_Time_Hours`).
2. **Unit Conversion** — Tempering time is converted from **seconds → hours**.
3. **Missing Value Handling** — Rows with missing `Hardness_HRC`, `Tempering_Temp_C`, or `C` are dropped. Remaining missing composition values are filled with `0.0`.
4. **Industrial Time Filter** — Raw lab data contains experiments up to **32 hours** of tempering (not commercially viable). Rows where `Tempering_Time_Hours > 4.0` are removed to ensure the model only learns practically efficient cycles.
5. **Austenitizing Temperature Derivation** — The raw dataset does not record austenitizing temperature. It is derived using the **Andrews (1965) Ac3 formula** with a +50°C superheat and small Gaussian noise (σ = 5°C):
   ```
   Ac3 = 910 - 203√C - 15.2·Ni + 44.7·Si + 104·V + 31.5·Mo
   Austenitizing_Temp = Ac3 + 50 + N(0, 5)
   ```
6. **Data Augmentation** — The original cleaned dataset is small (~few hundred rows). To ensure robust ML generalization, it is augmented with **Gaussian noise** to a target of **5,000 rows**:
   - Composition noise: σ = ±0.01% (clipped at 0)
   - Tempering temp noise: σ = ±5°C
   - Tempering time noise: σ = ±0.1 hours
   - Hardness noise: σ = ±1.0 HRC
   - Austenitizing temp is recalculated from synthetic composition to maintain physical consistency

---

## 🧠 Model Architecture

### Problem Formulation: Inverse Mapping

Classical heat treatment models work **forward** (composition + process → hardness). This app inverts the problem:

> **Given:** Alloy composition + Target Hardness  
> **Predict:** Optimal Austenitizing Temp, Tempering Temp, Tempering Time

### Training Architecture

| Component | Detail |
|---|---|
| **Algorithm** | `sklearn.RegressorChain` with `XGBRegressor` base |
| **Ensemble** | 5 independent models (seeds 0–4), averaged for prediction |
| **Target Order** | Austenitizing Temp → Tempering Temp → Tempering Time (chained to capture inter-target dependencies) |
| **Input Features** | `C, Mn, Si, Ni, Cr, Mo, V, P, S, Hardness_HRC` (10 features) |
| **Train/Test Split** | 80% / 20%, `random_state=42` |
| **Uncertainty** | ± standard deviation across 5 ensemble predictions |

### Why RegressorChain?
`RegressorChain` is used instead of simple MultiOutput regression because the three targets are **physically sequential and correlated**: higher austenitizing temps allow different dissolution kinetics, which affect what tempering is needed. Chaining lets each model condition on the previous target's prediction.

### Hyperparameter Optimization
Hyperparameters are tuned via **Optuna Bayesian optimization** (30 trials, 3-fold cross-validation, R² objective):

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 100 – 400 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.02 – 0.3 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |

---

## 📐 Metallurgical Formulas Used

### Carbon Equivalent (Dearden & O'Neill)
Used to assess weldability and hardenability:
```
Ceq = C + Mn/6 + (Cr + Mo + V)/5 + Ni/15
```

### Ideal Critical Diameter (Grossmann)
Indicates how deeply steel can harden under ideal quenching:
```
DI = 0.54√C × (1 + 4.37·Mn) × (1 + 2.16·Cr) × (1 + 3.0·Mo) × (1 + 0.363·Ni) × (1 + 1.5·Si)
```
*Converted from inches to mm (× 25.4)*

### Max Hardness Constraint (Burns-Moore-Archer)
Limits the target HRC slider dynamically based on Carbon content:
```
Max HRC = 35 + 50·C   (for C < 0.6%)
Max HRC = 65           (for C ≥ 0.6%)
```

### Quench Medium Selection Logic
```
(Cr + Mo) ≥ 4.5  → Air Quench
Target HRC ≤ 25  → Sand Cooling
C ≥ 0.45 or Ceq > 0.65 → Oil Quench
otherwise         → Water Quench
```

---

## 🗂️ Project Structure

```text
AI-Driven-Heat-Treatment-Optimizer/
├── data/
│   ├── raw/                         # Original Raiipa CSV dataset
│   └── processed/
│       └── heat_treatment_dataset.csv   # Cleaned + augmented training data (5000 rows)
├── models/
│   ├── heat_treatment_model_ensemble_0.pkl  # XGBoost ensemble model (seed 0)
│   ├── heat_treatment_model_ensemble_1.pkl  # ...
│   ├── heat_treatment_model_ensemble_2.pkl
│   ├── heat_treatment_model_ensemble_3.pkl
│   ├── heat_treatment_model_ensemble_4.pkl
│   └── heat_treatment_model_params.json     # Best Optuna hyperparameters
├── src/
│   ├── prepare_dataset.py           # Raw → processed data pipeline
│   ├── data_processing.py           # Data loader utility
│   ├── model.py                     # Model training + Optuna search + evaluation
│   ├── optimizer.py                 # Inference engine & tempering curve generation
│   ├── steel_grades.py              # AISI grade preset compositions
│   └── app.py                       # Streamlit web dashboard
├── tests/
│   └── test_optimizer.py            # Pytest unit tests
├── requirements.txt
└── README.md
```

---

## 🏗️ Retrain the Model

To retrain from scratch on the raw data:

```bash
# Step 1: Rebuild the processed dataset
python src/prepare_dataset.py

# Step 2: Run Optuna tuning + train ensemble + save models
python src/model.py
```

---

## 🧪 Running Tests

```bash
pytest tests/test_optimizer.py -v
```

---

## 🛣️ Roadmap / Suggested Improvements

- [ ] Expand training data to **tool steels, stainless steels, and high-carbon steels**
- [ ] Add **Jominy End-Quench** hardenability prediction
- [ ] Incorporate **microstructure prediction** (martensite %, retained austenite)
- [ ] Phase diagram integration for Ac1/Ac3 visualization
- [ ] Replace Gaussian augmentation with physics-based simulation data

---

## 🤝 Contributing

Pull requests and issues are very welcome! If you're a metallurgist, materials engineer, or ML practitioner and spot something that can be improved — please open an issue or PR on GitHub.

---

## 📜 License

This project is open-source under the **MIT License**.

---

## 👨‍🔬 Technical Context & References

- Andrews, K.W. (1965). *Empirical formulae for the calculation of some transformation temperatures.* JISI, 203, 721–727.
- Grossmann, M.A. (1942). *Principles of heat treatment.* ASM International.
- Dearden, J. & O'Neill, H. (1940). *A guide to the selection and welding of low alloy structural steels.* Trans. Inst. Weld. 3, 203–214.
- Burns, Moore & Archer (1938). *Hardness constraints in carbon steel heat treatment.*
