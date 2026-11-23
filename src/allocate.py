"""Budgeted allocation (Section 4.2), equity-impact frontier (4.3) and phased recruitment (4.4).

Listings 4-6 of the paper.
"""
import math
import numpy as np
import pandas as pd
import pulp
from config import CADRES, BANDS, SOLVER_TIME_LIMIT, cost_per_post, SANCTIONED

CBC = pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT)


def full_cost(f, cost=None):
    cost = cost or cost_per_post()
    return sum(f[f"{c}_gap"].sum() * cost[c] for c in CADRES)


def band_caps(act, req):
    """Posts available in each diminishing-returns band for one facility and cadre."""
    caps = []
    for lo, hi, _ in BANDS:
        a, b = math.ceil(lo * req), (math.ceil(hi * req) if hi < 1 else req)
        caps.append(max(0, b - max(a, act)) if act < b else 0)
    return caps


def allocate(f, posts):
    """Place posts[c] new staff in each cadre, deepest deficits first, ties broken by beds."""
    out = {c: np.zeros(len(f), int) for c in CADRES}
    for c in CADRES:
        n = min(int(posts[c]), int(f[f"{c}_gap"].sum()))
        if n <= 0:
            continue
        m = pulp.LpProblem("allocate", pulp.LpMaximize)
        terms, y = [], {}
        for k, r in enumerate(f.itertuples()):
            act, req = getattr(r, f"{c}_act"), getattr(r, f"{c}_req")
            for s, cap in enumerate(band_caps(act, req)):
                if cap > 0:
                    y[k, s] = pulp.LpVariable(f"y_{c}_{k}_{s}", 0, cap, cat="Integer")
                    terms.append(BANDS[s][2] * r.beds / req * y[k, s] + 1e-6 * y[k, s])
        m += pulp.lpSum(terms)
        m += pulp.lpSum(y.values()) == n
        m.solve(CBC)
        for (k, s), v in y.items():
            out[c][k] += int(round(v.value()))
    return out


def budget_scenarios(f, shares=(0.25, 0.5, 0.75)):
    """Allocation under budgets set as a share of the full-gap cost; each cadre closes the same share of its gap."""
    fc = full_cost(f)
    rows = []
    for sh in shares:
        x = allocate(f, {c: int(f[f"{c}_gap"].sum() * sh) for c in CADRES})
        row = {"budget_share": sh, "budget_cr": round(fc * sh, 1)}
        for c in CADRES:
            after = (f[f"{c}_act"] + x[c]) / f[f"{c}_req"]
            row[f"{c}_new_posts"] = int(x[c].sum())
            row[f"{c}_share_after"] = round((f[f"{c}_act"].sum() + x[c].sum()) / f[f"{c}_req"].sum(), 3)
            row[f"{c}_lowest_facility"] = round(after.min(), 3)
            row[f"{c}_facilities_below_25pct"] = int((after < .25).sum())
        rows.append(row)
    return pd.DataFrame(rows), fc


def frontier(f, budget_cr, target=0.5, floors=None, cost=None):
    """Epsilon-constraint frontier, solved exactly as a 0-1 knapsack (Example 5 of the paper)."""
    cost = cost or cost_per_post()
    floors = floors if floors is not None else [round(0.05 * i, 2) for i in range(10)]

    def need(r, c, share):
        return max(0, min(int(r[f"{c}_req"]), math.ceil(share * r[f"{c}_req"])) - int(r[f"{c}_act"]))

    pts = []
    for t in floors:
        base = {k: {c: need(r, c, t) for c in CADRES} for k, r in f.iterrows()}
        up = {k: {c: max(need(r, c, target), base[k][c]) for c in CADRES} for k, r in f.iterrows()}
        floor_cost = sum(cost[c] * base[k][c] for k in base for c in CADRES)
        if floor_cost > budget_cr:
            pts.append({"floor": t, "status": "Not affordable", "floor_cost_cr": round(floor_cost, 1)})
            continue
        delta = {k: sum(cost[c] * (up[k][c] - base[k][c]) for c in CADRES) for k in base}
        m = pulp.LpProblem("knapsack", pulp.LpMaximize)
        z = {k: pulp.LpVariable(f"z_{k}", cat="Binary") for k in base}
        m += pulp.lpSum(z.values()) + 1e-5 * pulp.lpSum(f.at[k, "beds"] * z[k] for k in z)
        m += pulp.lpSum(delta[k] * z[k] for k in z) <= budget_cr - floor_cost
        m.solve(CBC)
        sel = [k for k in z if round(z[k].value())]
        posts = {c: sum((up[k][c] if k in sel else base[k][c]) for k in base) for c in CADRES}
        pts.append({"floor": t, "status": pulp.LpStatus[m.status], "functional": len(sel),
                    "floor_cost_cr": round(floor_cost, 1), "beds_functional": int(f.loc[sel, "beds"].sum()),
                    **{f"{c}_posts": posts[c] for c in CADRES}})
    return pd.DataFrame(pts)


def phased(f, capacity, attrition, years, sanctioned=SANCTIONED, cost=None):
    """Year-by-year recruitment: remove leavers, hire up to capacity, place hires with allocate()."""
    cost = cost or cost_per_post()
    rows = []
    for name, cap in capacity.items():
        g = f.copy()
        for y in range(1, years + 1):
            rec = {"scenario": name, "year": y}
            for c in CADRES:
                g[f"{c}_act"] = g[f"{c}_act"] - (g[f"{c}_act"] * attrition).round().astype(int)
                g[f"{c}_gap"] = (g[f"{c}_req"] - g[f"{c}_act"]).clip(lower=0)
            x = allocate(g, {c: min(cap[c], int(g[f"{c}_gap"].sum())) for c in CADRES})
            for c in CADRES:
                g[f"{c}_act"] = g[f"{c}_act"] + x[c]
                g[f"{c}_gap"] = (g[f"{c}_req"] - g[f"{c}_act"]).clip(lower=0)
                filled = int(g[f"{c}_act"].sum())
                rec[f"{c}_hires"] = int(x[c].sum())
                rec[f"{c}_coverage"] = round(np.minimum(g[f"{c}_act"], g[f"{c}_req"]).sum() / g[f"{c}_req"].sum(), 3)
                rec[f"{c}_posts_to_create_cum"] = max(0, filled - sanctioned[c])
            rec["wage_bill_added_cr"] = round(sum((g[f"{c}_act"].sum() - f[f"{c}_act"].sum()) * cost[c] for c in CADRES), 1)
            rows.append(rec)
    return pd.DataFrame(rows)


def milestones(traj, share=1.0):
    """First year each cadre reaches `share` of its norm, by scenario (None if not within the horizon)."""
    out = {}
    for sc, d in traj.groupby("scenario", sort=False):
        out[sc] = {c: (int(d[d[f"{c}_coverage"] >= share].year.min()) if (d[f"{c}_coverage"] >= share).any() else None)
                   for c in CADRES}
    return out
