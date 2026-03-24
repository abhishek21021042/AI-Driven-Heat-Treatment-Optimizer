# 🔥 AI-Driven Heat Treatment Optimizer

**🔴  Demo link :** [Click here to launch the web application!](https://ai-driven-heat-treatment-optimizergit-efjuwo6tuo7m2a8zmvjviw.streamlit.app/)

A machine learning web dashboard that predicts the most **time- and energy-efficient heat treatment cycle** for low carbon steel alloys based on composition and desired hardness.

Instead of manual trial-and-error or static handbook charts, simply enter your alloy composition and target hardness — the AI instantly outputs the optimal **Austenitizing Temperature**, **Tempering Temperature**, **Tempering Time**, and **Quench Medium**.

> **⚠️ Note on Scope:** This model is specifically trained *only on low carbon steel* heat treatment data. Predictions for high-alloy, tool steels, or high-carbon steels will be highly extrapolative.

---

## ✨ Key Features

- **Static & Compact Dashboard:** A modern, scrolling-free single-screen UI built using Streamlit with completely custom CSS for a native app feel.
- **Grade Presets:** Quick-select dropdown for common steel grades (e.g. AISI 1020, 1040, 4140, 4340) which auto-fills the composition fields.
- **Hardenability Evaluation:** Instantly calculates the Carbon Equivalent (Ceq) and the Ideal Critical Diameter (DI) for the inputted alloy.
- **Dynamic Tempering Curve:** Interactive Plotly chart showing the predicted hardness response over a range of tempering temperatures.
- **Constraint Boundaries:** Enforces the Burns-Moore-Archer maximum hardness constraint dynamically based on the Carbon percentage.
- **Uncertainty Estimates:** Outputs ± margins of error for all continuous process parameters based on the variance across the underlying model ensemble.

---

## 🚀 Quick Start

Ensure you have Python 3.9+ installed.

```bash
# 1. Clone the repository and navigate to the directory
cd "heat treatment pridictor"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web application
python -m streamlit run src/app.py
```

The app will open instantly at **http://localhost:8501** in your default web browser.

---

## 🧠 Model Architecture

The core prediction engine uses a multi-target **Inverse Mapping** approach. Rather than forward-simulating and searching for an optimal point iteratively, the model has been trained to output process parameters directly from the desired output specifications.

| Component | Detail |
|---|---|
| **Algorithm** | XGBoost `RegressorChain` (5-model ensemble) |
| **Input Features** | %C, %Mn, %Si, %Ni, %Cr, %Mo, %V, %P, %S + Target Hardness (HRC) |
| **Targets Predicted** | Austenitizing Temp → Tempering Temp → Tempering Time → Quenchant |
| **Uncertainty** | ± Standard deviation calculated across the 5 models in the ensemble |

---

## 🗂️ Project Structure

```text
heat treatment pridictor/
├── data/
│   ├── raw/                     # Raw experimental datasets
│   └── processed/               # Cleaned + augmented training data
├── models/                      # Pickled ensemble models & best_params.json
├── src/
│   ├── prepare_dataset.py       # Data generation and cleaning pipeline
│   ├── data_processing.py       # Data load and split helpers
│   ├── model.py                 # Training script with Optuna hyperparameter tuning
│   ├── optimizer.py             # Inference engine & Tempering curve logic
│   ├── steel_grades.py          # AISI grade lookup dictionary
│   └── app.py                   # Streamlit web dashboard
├── tests/
│   └── test_optimizer.py        # Pytest suite
└── README.md                    # Project documentation
```

---

## 👨‍🔬 Technical Context

The training targets specifically focus on mitigating common issues found in classical heat treatment: minimizing tempering time to save energy, while ensuring the temperature stays within safe bounds to avoid temper embrittlement.

Austenitizing temperature is bounded dynamically by the Andrews (1965) Ac3 formula to ensure proper austenitization before quenching without inducing excessive grain growth.
