"""
tests/test_optimizer.py
Pytest unit tests for the Heat Treatment Optimizer.
Run:  pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

# ── Metallurgical formula tests (no model needed) ─────────────────────────────
from prepare_dataset import calculate_ac3

def test_ac3_pure_iron():
    """Ac3 of iron (0% C) should be close to 910°C."""
    row = {'C': 0.0, 'Ni': 0.0, 'Si': 0.0, 'V': 0.0, 'Mo': 0.0}
    assert abs(calculate_ac3(row) - 910) < 1

def test_ac3_decreases_with_carbon():
    """Higher carbon lowers Ac3."""
    r1 = calculate_ac3({'C': 0.2, 'Ni': 0.0, 'Si': 0.0, 'V': 0.0, 'Mo': 0.0})
    r2 = calculate_ac3({'C': 0.4, 'Ni': 0.0, 'Si': 0.0, 'V': 0.0, 'Mo': 0.0})
    assert r1 > r2

def test_ac3_nickel_lowers():
    """Nickel lowers Ac3 (austenite stabilizer)."""
    r_base = calculate_ac3({'C': 0.3, 'Ni': 0.0, 'Si': 0.0, 'V': 0.0, 'Mo': 0.0})
    r_ni   = calculate_ac3({'C': 0.3, 'Ni': 2.0, 'Si': 0.0, 'V': 0.0, 'Mo': 0.0})
    assert r_base > r_ni

# ── Quenchant logic tests (no model needed) ───────────────────────────────────
from optimizer import optimize_heat_treatment

@pytest.mark.skipif(not os.path.exists('models/heat_treatment_model_ensemble_0.pkl'),
                    reason="Ensemble models not trained yet — run python src/model.py first")
class TestOptimizer:
    BASE_COMP = {'C': 0.40, 'Mn': 0.80, 'Si': 0.25, 'Ni': 0.0, 'Cr': 1.0, 'Mo': 0.20, 'V': 0.0, 'P': 0.04, 'S': 0.04}

    def test_returns_expected_keys(self):
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        for key in ['Austenitizing_Temp_C', 'Tempering_Temp_C', 'Tempering_Time_Hours',
                    'Quench_Medium', 'Ceq', 'DI_mm']:
            assert key in res, f"Missing key: {key}"

    def test_temperatures_in_range(self):
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        assert 600 <= res['Austenitizing_Temp_C'] <= 1100, "Austenitizing out of range"
        assert 100 <= res['Tempering_Temp_C'] <= 750, "Tempering temp out of range"

    def test_time_positive(self):
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        assert res['Tempering_Time_Hours'] > 0

    def test_4140_oil_quench(self):
        """4140 steel (0.4C, 1Cr, 0.2Mo) must recommend Oil Quench."""
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        assert "Oil" in res['Quench_Medium']

    def test_high_alloy_air_quench(self):
        """H13-like composition (5% Cr + 1.35% Mo) should recommend Air Quench."""
        comp = {'C': 0.40, 'Mn': 0.40, 'Si': 1.00, 'Ni': 0.0, 'Cr': 5.25, 'Mo': 1.35, 'V': 1.0, 'P': 0.03, 'S': 0.03}
        res = optimize_heat_treatment(comp, 45.0)
        assert "Air" in res['Quench_Medium']

    def test_low_hardness_sand_cooling(self):
        """Target hardness <= 25 HRC should recommend Sand Cooling."""
        res = optimize_heat_treatment(self.BASE_COMP, 20.0)
        assert "Sand" in res['Quench_Medium']

    def test_ceq_positive(self):
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        assert res['Ceq'] > 0

    def test_di_positive(self):
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        assert res['DI_mm'] > 0

    def test_uncertainty_non_negative(self):
        res = optimize_heat_treatment(self.BASE_COMP, 45.0)
        assert res['Austenitizing_Temp_std'] >= 0
        assert res['Tempering_Temp_std'] >= 0
        assert res['Tempering_Time_std'] >= 0
