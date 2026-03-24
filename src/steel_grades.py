"""
src/steel_grades.py
Lookup table of common AISI/SAE steel grades with their nominal compositions.
Used to auto-fill the app composition fields.
"""

STEEL_GRADES = {
    "Custom (enter manually)": None,
    # --- Carbon Steels ---
    "AISI 1020 (low carbon)":    {"C": 0.20, "Mn": 0.45, "Si": 0.20, "Ni": 0.0,  "Cr": 0.0,  "Mo": 0.0,  "V": 0.0,  "P": 0.04, "S": 0.05},
    "AISI 1045 (medium carbon)": {"C": 0.45, "Mn": 0.75, "Si": 0.20, "Ni": 0.0,  "Cr": 0.0,  "Mo": 0.0,  "V": 0.0,  "P": 0.04, "S": 0.05},
    "AISI 1080 (high carbon)":   {"C": 0.80, "Mn": 0.75, "Si": 0.20, "Ni": 0.0,  "Cr": 0.0,  "Mo": 0.0,  "V": 0.0,  "P": 0.04, "S": 0.05},
    # --- alloy steels ---
    "AISI 4130 (Cr-Mo)":         {"C": 0.30, "Mn": 0.50, "Si": 0.25, "Ni": 0.0,  "Cr": 0.95, "Mo": 0.20, "V": 0.0,  "P": 0.04, "S": 0.04},
    "AISI 4140 (Cr-Mo)":         {"C": 0.40, "Mn": 0.90, "Si": 0.25, "Ni": 0.0,  "Cr": 1.05, "Mo": 0.20, "V": 0.0,  "P": 0.04, "S": 0.04},
    "AISI 4340 (Ni-Cr-Mo)":      {"C": 0.40, "Mn": 0.70, "Si": 0.25, "Ni": 1.80, "Cr": 0.80, "Mo": 0.25, "V": 0.0,  "P": 0.04, "S": 0.04},
    "AISI 8620 (case hardening)": {"C": 0.20, "Mn": 0.80, "Si": 0.25, "Ni": 0.55, "Cr": 0.50, "Mo": 0.20, "V": 0.0,  "P": 0.04, "S": 0.04},
    # --- Spring / Bearing steels ---
    "AISI 52100 (bearing)":       {"C": 1.00, "Mn": 0.35, "Si": 0.25, "Ni": 0.0,  "Cr": 1.50, "Mo": 0.0,  "V": 0.0,  "P": 0.03, "S": 0.03},
    "AISI 5160 (spring)":         {"C": 0.60, "Mn": 0.88, "Si": 0.25, "Ni": 0.0,  "Cr": 0.80, "Mo": 0.0,  "V": 0.0,  "P": 0.04, "S": 0.04},
    "AISI 9260 (Si spring)":      {"C": 0.60, "Mn": 0.88, "Si": 1.80, "Ni": 0.0,  "Cr": 0.0,  "Mo": 0.0,  "V": 0.0,  "P": 0.04, "S": 0.04},
    # --- Tool steels ---
    "AISI H13 (hot work)":        {"C": 0.40, "Mn": 0.40, "Si": 1.00, "Ni": 0.0,  "Cr": 5.25, "Mo": 1.35, "V": 1.00, "P": 0.03, "S": 0.03},
    "AISI D2 (cold work)":        {"C": 1.55, "Mn": 0.40, "Si": 0.40, "Ni": 0.0,  "Cr":11.50, "Mo": 0.80, "V": 0.80, "P": 0.03, "S": 0.03},
    "AISI O1 (oil hardening)":    {"C": 0.90, "Mn": 1.20, "Si": 0.30, "Ni": 0.0,  "Cr": 0.50, "Mo": 0.0,  "V": 0.20, "P": 0.03, "S": 0.03},
    "AISI M2 (HSS)":              {"C": 0.85, "Mn": 0.30, "Si": 0.30, "Ni": 0.0,  "Cr": 4.20, "Mo": 5.00, "V": 1.90, "P": 0.03, "S": 0.03},
}
