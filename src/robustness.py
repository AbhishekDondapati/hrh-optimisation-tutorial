"""Monte Carlo robustness of the phased-recruitment results (nurses and paramedics).
Capacity each year ~ U(0.75, 1.25) x scenario value; attrition per run ~ U(2%, 4%).
Allocation within a year uses the exact greedy solution of the banded allocation model
(a fractional knapsack with unit costs and integral caps, so greedy is optimal)."""
import math, sys, json, numpy as np, pandas as pd
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / "data" / "facilities.csv"
f = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA)
BANDS = [(0.0, 0.4, 1.0), (0.4, 0.7, 0.5), (0.7, 1.0, 0.2)]
CAP = {"Low": {"N": 270, "P": 200}, "Base": {"N": 910, "P": 440}, "High": {"N": 1860, "P": 1010}}
req = {c: f[f"{c}_req"].values for c in "NP"}; beds = f.beds.values

def hire(act, R, n):
    items = []
    for k in range(len(act)):
        for lo, hi, v in BANDS:
            a, b = math.ceil(lo * R[k]), (math.ceil(hi * R[k]) if hi < 1 else R[k])
            cap = max(0, b - max(a, act[k])) if act[k] < b else 0
            if cap: items.append((-(v * beds[k] / R[k]), k, cap))
    items.sort()
    act = act.copy(); left = n
    for _, k, cap in items:
        if left <= 0: break
        q = min(cap, left); act[k] += q; left -= q
    return act

def run(c, cap, alpha, years=15, rng=None, noise=True):
    act = f[f"{c}_act"].values.astype(int).copy(); R = req[c]; out = []
    for y in range(years):
        act = act - np.round(act * alpha).astype(int)
        gap = int(np.clip(R - act, 0, None).sum())
        h = cap * (rng.uniform(0.75, 1.25) if noise else 1)
        act = hire(act, R, min(int(round(h)), gap))
        out.append(np.minimum(act, R).sum() / R.sum())
    return out

def first(path, s=1.0):
    for i, v in enumerate(path):
        if v >= s - 1e-9: return i + 1
    return None

res = {"deterministic": {}, "mc": {}}
for sc, cp in CAP.items():
    res["deterministic"][sc] = {c: first(run(c, cp[c], 0.03, noise=False)) for c in "NP"}
rng = np.random.default_rng(2027); NRUN = int(sys.argv[2]) if len(sys.argv) > 2 else 500  # usage: python src/robustness.py [facilities.csv] [runs]
for sc in ("Base", "High"):
    out = {}
    for c in "NP":
        yrs, y15 = [], []
        for _ in range(NRUN):
            p = run(c, CAP[sc][c], rng.uniform(0.02, 0.04), rng=rng)
            yrs.append(first(p)); y15.append(p[-1])
        hit = [y for y in yrs if y is not None]
        out[c] = {"share_reaching_norm_15y": round(len(hit) / NRUN, 3),
                  "median_year": (float(np.median(hit)) if hit else None),
                  "p05_year": (float(np.percentile(hit, 5)) if hit else None),
                  "p95_year": (float(np.percentile(hit, 95)) if hit else None),
                  "coverage_y15_p05": round(float(np.percentile(y15, 5)), 3),
                  "coverage_y15_median": round(float(np.median(y15)), 3),
                  "within_10y": round(sum(1 for y in yrs if y is not None and y <= 10) / NRUN, 3)}
    res["mc"][sc] = out
print(json.dumps(res, indent=1))
