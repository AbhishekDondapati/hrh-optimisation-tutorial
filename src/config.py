"""Parameters used in the guided example (Section 5 of the paper).

Change these to apply the workflow to another network.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CADRES = ["D", "N", "P"]                      # doctors, staff nurses, paramedics
CADRE_NAMES = {"D": "Doctors", "N": "Staff Nurses", "P": "Paramedics"}

# Specialties that together make a facility "referral-ready" (Section 3.2)
TRIAD = ["Obstetrics & Gynaecology", "Anaesthesia", "Paediatrics"]

# Cost per post: entry basic pay (Rs per month) x gross-up factor x 12 (Section 5.5, endnote 2)
BASIC_PAY = {"D": 58850, "N": 36750, "P": 32810}
GROSS_UP = 1.45

def cost_per_post(gross_up=GROSS_UP):
    """Annual cost per post in Rs crore."""
    return {c: BASIC_PAY[c] * gross_up * 12 / 1e7 for c in CADRES}

# Diminishing-returns bands for the allocation model (Section 4.2): (from, to, weight)
BANDS = [(0.0, 0.4, 1.0), (0.4, 0.7, 0.5), (0.7, 1.0, 0.2)]

# Redeployment model (Section 4.1)
PRIORITY = {"triad": 5.0, "empty_dept": 3.0, "other": 1.0, "new_triad": 3.0, "new_other": 1.0}
MOVE_COST = 0.05        # kappa: disruption cost per move
CROSS_DISTRICT_COST = 0.05  # kappa': extra cost when a move crosses a district boundary

# Phased recruitment (Section 4.4): staff who can join per year, by scenario
CAPACITY = {
    "Low (historic intake)": {"D": 160, "N": 270, "P": 200},
    "Base (recent batches)": {"D": 540, "N": 910, "P": 440},
    "High (dedicated drive)": {"D": 780, "N": 1860, "P": 1010},
}
ATTRITION = 0.03        # share of staff in post leaving each year
YEARS = 15

# Sanctioned posts in the simulated network, used to compute posts that must be newly created
SANCTIONED = {"D": 3254, "N": 3252, "P": 1402}

SOLVER_TIME_LIMIT = 120  # seconds per integer programme
