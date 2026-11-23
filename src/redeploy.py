"""Redeployment as an integer transportation problem (Section 4.1, Listing 3)."""
from collections import defaultdict
import pandas as pd
import pulp
from config import TRIAD, PRIORITY, MOVE_COST, CROSS_DISTRICT_COST
from data import spec_category

# Designations each department accepts
COMPAT = {
    "Radiology & Pathology": ["Radiologist", "Pathologist"],
    "RMO": ["RMO", "General Duty Medical Officer (Non-Specialist)"],
    "Dialysis Centre": ["General Duty Medical Officer (Non-Specialist)"],
}
DESIG = {"General Medicine": "General Medicine Specialist", "General Surgery": "General Surgery Specialist",
         "Dental Surgery": "Dental Surgeon", "Obstetrics & Gynaecology": "OBG specialist",
         "Anaesthesia": "Anesthesiologist", "Paediatrics": "Paediatrician", "Orthopaedics": "Orthopaedician",
         "Ophthalmology": "Ophthalmology Specialist", "ENT": "ENT specialist", "Dermatology": "Dermatologist",
         "Psychiatry": "Psychiatrist", "Pulmonology": "Pulmonary Medicine Specialist",
         "Microbiology": "Microbiologist", "Forensic Medicine": "Forensic Medicine specialist"}


def accepts(dept):
    return COMPAT.get(dept, [DESIG.get(dept)])


def supply_demand(f, roster, misplaced, norms, weights=PRIORITY):
    """Build supply nodes (movable specialists) and demand nodes (department shortfalls)."""
    sup = roster[roster.excess > 0][["fid", "desig", "excess"]].rename(columns={"excess": "n"}).assign(kind="excess")
    sup = pd.concat([sup, misplaced.rename(columns={"avail": "n"})[["fid", "desig", "n"]].assign(kind="misplaced")])
    sup = sup.groupby(["fid", "desig"], as_index=False).agg(n=("n", "sum"), kind=("kind", "first"))
    sup["n"] = sup.n.astype(int)

    dem = roster[roster.short > 0][["fid", "dept", "req", "avail", "short"]].copy()
    dem["w"] = weights["other"]
    dem.loc[dem.avail == 0, "w"] = weights["empty_dept"]
    dem.loc[dem.dept.isin(TRIAD), "w"] = weights["triad"]
    dem["new"] = False
    # facilities with no specialist roster: demand is the full specialist norm of their category
    rows = []
    for _, r in f[~f.fid.isin(roster.fid.unique())].iterrows():
        for _, n in norms[norms.cat == spec_category(r.tier)].iterrows():
            rows.append({"fid": r.fid, "dept": n.dept, "req": int(n.req), "avail": 0, "short": int(n.req),
                         "w": weights["new_triad"] if n.dept in TRIAD else weights["new_other"], "new": True})
    dem = pd.concat([dem, pd.DataFrame(rows)], ignore_index=True)
    return sup.reset_index(drop=True), dem


def solve(f, sup, dem, scope="zone", move_cost=MOVE_COST, xdist_cost=CROSS_DISTRICT_COST):
    """Solve the transportation model within a transfer scope: district, zone, multizone or network."""
    geo = f.set_index("fid")[["district", "zone", "multizone"]]
    key = {"district": "district", "zone": "zone", "multizone": "multizone", "network": None}[scope]
    G = geo.to_dict("index")
    acc = {d: set(accepts(d)) for d in dem.dept.unique()}
    arcs = []
    for i, sf, sd in sup[["fid", "desig"]].itertuples():
        for j, df_, dd, w in dem[["fid", "dept", "w"]].itertuples():
            if sf == df_ or sd not in acc[dd]:
                continue
            if key and G[sf][key] != G[df_][key]:
                continue
            cross = G[sf]["district"] != G[df_]["district"]
            arcs.append((i, j, w - move_cost - (xdist_cost if cross else 0)))

    m = pulp.LpProblem("redeploy", pulp.LpMaximize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0, None, cat="Integer") for i, j, _ in arcs}
    m += pulp.lpSum(v * x[i, j] for i, j, v in arcs)
    by_i, by_j = defaultdict(list), defaultdict(list)
    for a in x:
        by_i[a[0]].append(x[a]); by_j[a[1]].append(x[a])
    for i, v in by_i.items():
        m += pulp.lpSum(v) <= int(sup.at[i, "n"])        # cannot move more than the surplus
    for j, v in by_j.items():
        m += pulp.lpSum(v) <= int(dem.at[j, "short"])    # cannot fill more than the gap
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status]

    name = f.set_index("fid").facility
    label = {PRIORITY["triad"]: "Referral-specialist gap", PRIORITY["empty_dept"]: "Department with no specialist",
             PRIORITY["other"]: "Other shortfall"}
    rows = []
    for (i, j), var in x.items():
        q = int(round(var.value() or 0))
        if not q:
            continue
        s, d = sup.loc[i], dem.loc[j]
        if d.new:
            prio = "New facility - referral" if d.dept in TRIAD else "New facility - other"
        else:
            prio = "Referral-specialist gap" if d.dept in TRIAD else ("Department with no specialist" if d.avail == 0 else "Other shortfall")
        rows.append({"from_facility": name[s.fid], "from_district": geo.at[s.fid, "district"], "designation": s.desig,
                     "source": s.kind, "to_facility": name[d.fid], "to_district": geo.at[d.fid, "district"],
                     "to_department": d.dept, "posts": q, "priority": prio, "from_fid": s.fid, "to_fid": d.fid})
    return pd.DataFrame(rows), status


def summarise(moves, sup, dem):
    if moves.empty:
        return {"moves": 0}
    p = moves.groupby("priority").posts.sum()
    return {"moves": int(moves.posts.sum()),
            "referral_gaps_closed": int(p.get("Referral-specialist gap", 0)),
            "referral_gaps_total": int(dem[(~dem.new) & dem.dept.isin(TRIAD)].short.sum()),
            "empty_departments_filled": int(p.get("Department with no specialist", 0)),
            "other_existing": int(p.get("Other shortfall", 0)),
            "new_referral_filled": int(p.get("New facility - referral", 0)),
            "new_referral_total": int(dem[dem.new & dem.dept.isin(TRIAD)].short.sum()),
            "new_other": int(p.get("New facility - other", 0)),
            "cross_district": int(moves[moves.from_district != moves.to_district].posts.sum()),
            "donors": int(moves.from_fid.nunique()), "recipients": int(moves.to_fid.nunique())}


def apply_moves(f, moves):
    """Update doctors in post after the moves."""
    f = f.copy()
    if moves is not None and len(moves):
        for fid, q in moves.groupby("from_fid").posts.sum().items():
            f.loc[f.fid == fid, "D_act"] -= q
        for fid, q in moves.groupby("to_fid").posts.sum().items():
            f.loc[f.fid == fid, "D_act"] += q
    from data import add_gaps
    return add_gaps(f)
