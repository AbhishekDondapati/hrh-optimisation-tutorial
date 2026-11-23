"""Load the input files and derive adherence and gaps (Section 2.1, Listing 1)."""
import pandas as pd
from config import DATA, CADRES


def add_gaps(f):
    """Add gap and adherence columns for every cadre."""
    f = f.copy()
    for c in CADRES:
        f[f"{c}_gap"] = (f[f"{c}_req"] - f[f"{c}_act"]).clip(lower=0)
        f[f"{c}_pct"] = f[f"{c}_act"] / f[f"{c}_req"]
    return f


def load():
    """Return (facilities, roster, misplaced, norms).

    facilities : one row per facility, with staff in post (_act) and IPHS requirement (_req)
    roster     : one row per facility x department of the specialist norm, with specialists in post
                 (only facilities that have a specialist roster)
    misplaced  : specialists posted to a department their facility's norm does not include
    norms      : specialist norm by facility category and department
    """
    f = add_gaps(pd.read_csv(DATA / "facilities.csv"))
    roster = pd.read_csv(DATA / "roster.csv")
    roster["short"] = (roster.req - roster.avail).clip(lower=0)
    roster["excess"] = (roster.avail - roster.req).clip(lower=0)
    roster.loc[roster.cat == "DH", "excess"] = 0     # district hospitals keep their surplus (training, referral load)
    misplaced = pd.read_csv(DATA / "misplaced.csv")
    norms = pd.read_csv(DATA / "specialist_norms.csv")
    return f, roster, misplaced, norms


def spec_category(tier):
    """Map a facility tier (e.g. AH-150) to its specialist-norm category (e.g. AH)."""
    if tier.startswith("AH"):
        return "AH"
    if tier.startswith("DH"):
        return "DH"
    return tier
