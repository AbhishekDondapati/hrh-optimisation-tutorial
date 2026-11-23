"""Inequality diagnostics and the capability test (Section 3, Listing 2)."""
import numpy as np
import pandas as pd
from config import CADRES, TRIAD


def wgini(x, w):
    """Gini coefficient of adherence x, each facility weighted by its required posts w."""
    o = np.argsort(np.asarray(x))
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    W = np.r_[0, np.cumsum(w) / w.sum()]
    S = np.r_[0, np.cumsum(x * w) / (x * w).sum()]
    return 1 - np.sum((W[1:] - W[:-1]) * (S[1:] + S[:-1]))


def theil(df, a, r, group):
    """Theil-T of staff per required post, with its exact between/within-group split.

    Returns (total, between, within).
    """
    mu = df[a].sum() / df[r].sum()

    def T(sub):
        m = sub[a].sum() / sub[r].sum()
        s = sub[sub[a] > 0]                       # 0 ln 0 = 0
        return (s[a] / sub[a].sum() * np.log((s[a] / s[r]) / m)).sum()

    total = T(df)
    between = 0.0
    for _, sub in df.groupby(group):
        if sub[a].sum() == 0:
            continue
        m = sub[a].sum() / sub[r].sum()
        between += sub[a].sum() / df[a].sum() * np.log(m / mu)
    return total, between, total - between


def bands(df):
    """Number of facilities in each adherence band, by cadre."""
    edges = [0, .25, .5, .75, 1.0001, 99]
    labels = ["<25%", "25-50%", "50-75%", "75-100%", ">=100%"]
    return {c: pd.cut(df[f"{c}_pct"], edges, labels=labels, right=False)
            .value_counts().reindex(labels).astype(int).to_dict() for c in CADRES}


def referral_ready(f, roster):
    """True for each facility with at least one specialist in post in every triad specialty."""
    t = (roster[roster.dept.isin(TRIAD)]
         .pivot_table(index="fid", columns="dept", values="avail", aggfunc="sum")
         .reindex(columns=TRIAD).reindex(f.fid).fillna(0))
    return (t >= 1).all(axis=1)


def run(f, roster):
    ex = f[f.status == "Existing"]
    out = {
        "bands_existing": bands(ex),
        "gini_existing": {c: round(float(wgini(ex[f"{c}_pct"], ex[f"{c}_req"])), 3) for c in CADRES},
        "gini_all": {c: round(float(wgini(f[f"{c}_pct"], f[f"{c}_req"])), 3) for c in CADRES},
        "theil_between_share_pct": {},
    }
    for c in CADRES:
        out["theil_between_share_pct"][c] = {}
        for grp in ["zone", "district", "tier"]:
            t, b, _ = theil(ex, f"{c}_act", f"{c}_req", grp)
            out["theil_between_share_pct"][c][grp] = round(float(100 * b / t), 1)
        out["theil_between_share_pct"][c]["total_T"] = round(float(t), 3)
    ready = referral_ready(f, roster)
    st = f.set_index("fid").status
    out["referral_ready"] = {"existing": int(ready[st == "Existing"].sum()), "of_existing": int((st == "Existing").sum()),
                             "new": int(ready[st == "New"].sum()), "of_new": int((st == "New").sum())}
    tri = roster[roster.dept.isin(TRIAD)].groupby("dept")[["avail", "short", "excess"]].sum()
    out["triad"] = tri.astype(int).to_dict("index")
    return out
