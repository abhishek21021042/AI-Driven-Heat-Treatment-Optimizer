"""
src/app.py — Heat Treatment Optimizer (Phase 2 UI)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    from optimizer import optimize_heat_treatment, predict_tempering_curve
    from steel_grades import STEEL_GRADES
except ImportError:
    from .optimizer import optimize_heat_treatment, predict_tempering_curve
    from .steel_grades import STEEL_GRADES

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heat Treatment Optimizer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d1117; color: #c9d1d9; }
h1,h2,h3 { color: #58a6ff; }
.metric-card {
    background: linear-gradient(145deg,#1f2428,#24292e);
    border-radius: 8px; padding: 10px; text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4); border: 1px solid #30363d;
    transition: transform .2s ease, border-color .2s ease;
    margin-bottom: 10px;
}
.metric-card:hover { transform: translateY(-3px); border-color: #58a6ff; }
.metric-title  { font-size:.85rem; color:#8b949e; margin-bottom:2px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value  { font-size:1.6rem; font-weight:700; line-height: 1.2; }
.metric-unit   { font-size:.75rem; color:#8b949e; }
.metric-std    { font-size:.7rem; color:#6e7681; margin-top:2px; }
.info-card {
    background:#161b22; border-radius:10px; padding:14px; text-align:center;
    border:1px solid #30363d; margin-bottom:10px;
}
.info-title { font-size:.85rem; color:#8b949e; }
.info-value { font-size:1.5rem; font-weight:600; color:#c9d1d9; }
.stButton>button {
    background: linear-gradient(90deg,#238636,#2ea043); color:white;
    border:none; border-radius:8px; padding:.5rem 1.8rem; font-weight:bold;
    transition: all .3s;
}
.stButton>button:hover {
    background:linear-gradient(90deg,#2ea043,#3fb950);
    box-shadow:0 0 15px rgba(46,160,67,.4);
}
</style>
""", unsafe_allow_html=True)

# ── Header App Logo ───────────────────────────────────────────────────────────
st.markdown("<h3 style='margin-top: -15px; margin-bottom: 5px; color: #58a6ff;'>🔥 AI-Driven Heat Treatment Optimizer</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.85rem; color: #8b949e; margin-bottom: 25px; margin-top: -5px;'><em>⚠️ Note: This model is trained only on low carbon steel heat treatment data.</em></p>", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1.3], gap="large")

with col_input:
    # C1: Grade selector
    st.markdown("### ⚙️ Steel Grade")
    selected_grade = st.selectbox("Quick-select a known grade (auto-fills composition)", list(STEEL_GRADES.keys()))

    if STEEL_GRADES[selected_grade]:
        preset = STEEL_GRADES[selected_grade]
    else:
        preset = {}

    st.markdown("### 🛠️ Alloy Composition (%wt)")
    c1, c2 = st.columns(2)
    with c1:
        c_pct  = st.number_input("Carbon (C)",      min_value=0.0, max_value=2.0,  value=preset.get("C",  0.40), step=0.01)
        mn_pct = st.number_input("Manganese (Mn)",  min_value=0.0, max_value=2.0,  value=preset.get("Mn", 0.80), step=0.05)
        si_pct = st.number_input("Silicon (Si)",    min_value=0.0, max_value=2.0,  value=preset.get("Si", 0.20), step=0.05)
        ni_pct = st.number_input("Nickel (Ni)",     min_value=0.0, max_value=5.0,  value=preset.get("Ni", 0.00), step=0.10)
    with c2:
        cr_pct = st.number_input("Chromium (Cr)",   min_value=0.0, max_value=12.0, value=preset.get("Cr", 1.00), step=0.10)
        mo_pct = st.number_input("Molybdenum (Mo)", min_value=0.0, max_value=5.0,  value=preset.get("Mo", 0.20), step=0.05)
        v_pct  = st.number_input("Vanadium (V)",    min_value=0.0, max_value=2.0,  value=preset.get("V",  0.00), step=0.01)
        p_pct  = st.number_input("Phosphorus (P)",  min_value=0.0, max_value=0.10, value=preset.get("P",  0.04), step=0.005, format="%.4f")
        s_pct  = st.number_input("Sulfur (S)",      min_value=0.0, max_value=0.10, value=preset.get("S",  0.04), step=0.005, format="%.4f")

    # Burns-Moore-Archer constraint on HRC
    max_hrc = 65.0 if c_pct >= 0.60 else float(np.round(35.0 + 50.0 * c_pct, 1))
    max_hrc = max(25.0, max_hrc)

    st.markdown("---")
    st.markdown("### 🎯 Desired Hardness")
    st.caption(f"Burns-Moore-Archer max for **{c_pct:.2f}% C** → **{max_hrc} HRC**")

    if 'target_slider' not in st.session_state:
        st.session_state['target_slider'] = min(45.0, max_hrc)
    if st.session_state['target_slider'] > max_hrc:
        st.session_state['target_slider'] = max_hrc

    target_hardness = st.slider("Target Hardness (HRC)", min_value=20.0, max_value=max_hrc, key='target_slider', step=0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Optimize Heat Treatment")

# ── Output Column ──────────────────────────────────────────────────────────────
with col_output:
    st.markdown("### 🔬 Optimal Process Parameters")

    # Run prediction on button press
    if predict_btn:
        comp_dict = {'C': c_pct, 'Mn': mn_pct, 'Si': si_pct,
                     'Ni': ni_pct, 'Cr': cr_pct, 'Mo': mo_pct,
                     'V': v_pct, 'P': p_pct, 'S': s_pct}
        with st.spinner("Ensemble AI optimizing..."):
            try:
                result = optimize_heat_treatment(comp_dict, target_hardness)
                st.session_state['predictions'] = result
            except Exception as e:
                st.error(f"Prediction error: {e}")

    if st.session_state.get('predictions'):
        r = st.session_state['predictions']

        # Main 4-card grid
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🔥 Austenitizing Temp</div>
                <div class="metric-value" style="color:#ff7b72;">{r['Austenitizing_Temp_C']:.0f}<span class="metric-unit"> °C</span></div>
                <div class="metric-std">± {r['Austenitizing_Temp_std']:.0f} °C uncertainty</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">⏱️ Tempering Time</div>
                <div class="metric-value" style="color:#d2a8ff;">{r['Tempering_Time_Hours']:.2f}<span class="metric-unit"> h</span></div>
                <div class="metric-std">± {r['Tempering_Time_std']:.2f} h uncertainty</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">❄️ Tempering Temp</div>
                <div class="metric-value" style="color:#79c0ff;">{r['Tempering_Temp_C']:.0f}<span class="metric-unit"> °C</span></div>
                <div class="metric-std">± {r['Tempering_Temp_std']:.0f} °C uncertainty</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🌊 Quenchant</div>
                <div class="metric-value" style="color:#a5d6ff;font-size:1.6rem;">{r['Quench_Medium']}</div>
                <div class="metric-std">&nbsp;</div>
            </div>""", unsafe_allow_html=True)

        # C3: Ceq + DI info cards
        st.markdown("**📊 Hardenability Indicators**")
        hi1, hi2 = st.columns(2)
        with hi1:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Carbon Equivalent (Ceq)</div>
                <div class="info-value">{r['Ceq']:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with hi2:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Ideal Critical Diameter DI</div>
                <div class="info-value">{r['DI_mm']:.0f} mm</div>
            </div>""", unsafe_allow_html=True)

        # C2: Tempering Curve
        st.markdown("<h4 style='margin-top:20px; color:#8b949e;'>📈 Predicted Tempering Curve</h4>", unsafe_allow_html=True)
        comp_dict = {'C': c_pct, 'Mn': mn_pct, 'Si': si_pct,
                     'Ni': ni_pct, 'Cr': cr_pct, 'Mo': mo_pct,
                     'V': v_pct, 'P': p_pct, 'S': s_pct}
        with st.spinner("Generating tempering curve..."):
            curve_df = predict_tempering_curve(comp_dict)

        if not curve_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_df['Tempering_Temp_C'], y=curve_df['Hardness_HRC'],
                mode='lines+markers', name='Tempering Curve',
                line=dict(color='#79c0ff', width=2.5),
                marker=dict(size=6, color='#58a6ff')
            ))
            fig.add_vline(x=r['Tempering_Temp_C'], line_dash="dash",
                          line_color="#ff7b72", annotation_text=f"Target {target_hardness} HRC",
                          annotation_position="top right")
            fig.update_layout(
                paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
                font_color='#c9d1d9',
                xaxis=dict(title='Tempering Temperature (°C)', gridcolor='#30363d'),
                yaxis=dict(title='Final Hardness (HRC)', gridcolor='#30363d'),
                margin=dict(l=10, r=10, t=30, b=10), height=320
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Select a steel grade (or enter custom composition), set your Target Hardness, then click **Optimize**.")
        # Placeholder cards
        for label in ["🔥 Austenitizing Temp", "❄️ Tempering Temp", "⏱️ Tempering Time", "🌊 Quenchant"]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">{label}</div><div class="metric-value">---</div></div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#8b949e;font-size:.85rem;'>AI-Driven Heat Treatment Optimizer · XGBoost Ensemble · Streamlit</p>", unsafe_allow_html=True)
