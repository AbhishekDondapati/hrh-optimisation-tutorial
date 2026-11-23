"""Reproduce every result in Section 5 and Appendix A of the paper.

Usage:  python src/run_all.py            main results (about 1-3 minutes)
        python src/run_all.py --sens     also the sensitivity analyses of Appendix A
"""
import json
import sys
import time
import config as C
from data import load
import diagnostics, redeploy, allocate


def main(sens=False):
    t0 = time.time()
    C.RESULTS.mkdir(exist_ok=True)
    f, roster, misplaced, norms = load()
    R = {"network": {"facilities": len(f), "existing": int((f.status == "Existing").sum()),
                     "new": int((f.status == "New").sum()), "districts": f.district.nunique(), "zones": f.zone.nunique()},
         "totals": {c: {"in_post": int(f[f"{c}_act"].sum()), "required": int(f[f"{c}_req"].sum()),
                        "gap": int(f[f"{c}_gap"].sum())} for c in C.CADRES}}

    # Section 5.1: diagnostics and capability test
    R["diagnostics"] = diagnostics.run(f, roster)

    # Section 5.2: redeployment by transfer unit
    sup, dem = redeploy.supply_demand(f, roster, misplaced, norms)
    R["redeployment_inputs"] = {"supply": int(sup.n.sum()), "misplaced": int(misplaced.avail.sum()),
                                "demand": int(dem.short.sum()), "demand_existing": int(dem[~dem.new].short.sum())}
    R["redeployment"] = {}
    for scope in ["district", "zone", "multizone", "network"]:
        mv, status = redeploy.solve(f, sup, dem, scope)
        assert status == "Optimal", (scope, status)
        R["redeployment"][scope] = redeploy.summarise(mv, sup, dem)
        if scope == "zone":
            moves = mv
    moves.drop(columns=["from_fid", "to_fid"]).to_csv(C.RESULTS / "zone_moves.csv", index=False)
    fa = redeploy.apply_moves(f, moves)
    ready = diagnostics.referral_ready(fa, _roster_after(roster, moves))
    st = fa.set_index("fid").status
    R["referral_ready_after_zone_moves"] = {"existing": int(ready[st == "Existing"].sum()), "new": int(ready[st == "New"].sum())}

    # Section 5.3: budgeted allocation
    b2, fc = allocate.budget_scenarios(fa)
    R["full_gap_cost_cr"] = round(fc, 1)
    R["budget_scenarios"] = b2.to_dict("records")

    # Section 5.3: equity-impact frontier
    R["frontier_25pct_budget_target50"] = allocate.frontier(fa, fc * 0.25, 0.5).to_dict("records")
    R["frontier_50pct_budget_target75"] = allocate.frontier(fa, fc * 0.50, 0.75).to_dict("records")

    # Section 5.4: phased recruitment
    traj = allocate.phased(fa, C.CAPACITY, C.ATTRITION, C.YEARS)
    traj.to_csv(C.RESULTS / "phased_recruitment.csv", index=False)
    R["year_norm_reached"] = allocate.milestones(traj, 1.0)
    R["year_half_norm_reached"] = allocate.milestones(traj, 0.5)

    if sens:
        R["sensitivity"] = sensitivity(f, fa, sup, dem, fc)

    with open(C.RESULTS / "results.json", "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    report(R)
    print(f"\nAll results written to {C.RESULTS} in {time.time() - t0:.0f} s")


def _roster_after(roster, moves):
    """Specialist roster after the moves (needed for the readiness test)."""
    r = roster[["fid", "dept", "avail"]].copy()
    tri = {"OBG specialist": "Obstetrics & Gynaecology", "Anesthesiologist": "Anaesthesia", "Paediatrician": "Paediatrics"}
    adds = []
    for _, m in moves.iterrows():
        if m.designation in tri:
            dept = tri[m.designation]
            r.loc[(r.fid == m.from_fid) & (r.dept == dept), "avail"] -= m.posts
            if m.to_department == dept:
                adds.append({"fid": m.to_fid, "dept": dept, "avail": m.posts})
    import pandas as pd
    return pd.concat([r, pd.DataFrame(adds)], ignore_index=True)


def sensitivity(f, fa, sup, dem, fc):
    """Appendix A, Table A1."""
    out = {}
    d_eq = dem.copy(); d_eq["w"] = 1.0
    mv, _ = redeploy.solve(f, sup, d_eq, "zone")
    s = redeploy.summarise(mv, sup, dem)
    out["equal_weights"] = {k: s[k] for k in ("moves", "referral_gaps_closed", "empty_departments_filled")}
    mv, _ = redeploy.solve(f, sup, dem, "zone", move_cost=0.5, xdist_cost=0.5)
    out["disruption_cost_x10"] = redeploy.summarise(mv, sup, dem)
    out["full_cost_gross_up"] = {k: round(allocate.full_cost(fa, C.cost_per_post(k)), 1) for k in (1.30, 1.60)}
    r = allocate.frontier(fa, fc * 0.25, 0.6)
    out["frontier_target60"] = {str(p.floor): (None if p.status != "Optimal" else int(p.functional)) for p in r.itertuples()}
    base = {"Base (recent batches)": C.CAPACITY["Base (recent batches)"]}
    for a in (0.02, 0.04):
        traj = allocate.phased(fa, base, a, C.YEARS)
        out[f"attrition_{a}"] = {"year_norm_reached": allocate.milestones(traj, 1.0)["Base (recent batches)"],
                                 "nurse_coverage_year15": float(traj.N_coverage.iloc[-1])}
    return out


def report(R):
    t = R["totals"]
    print("Network:", R["network"])
    print("Totals (in post / required / gap):", {c: (v["in_post"], v["required"], v["gap"]) for c, v in t.items()})
    print("Gini, existing facilities:", R["diagnostics"]["gini_existing"])
    print("Referral-ready before moves:", R["diagnostics"]["referral_ready"])
    print("\nTable 2 - redeployment by transfer unit")
    for sc, v in R["redeployment"].items():
        print(f"  {sc:10s} moves {v['moves']:4d}  referral gaps {v['referral_gaps_closed']}/{v['referral_gaps_total']}"
              f"  empty depts {v['empty_departments_filled']:3d}  new referral {v['new_referral_filled']}/{v['new_referral_total']}"
              f"  cross-district {v['cross_district']}")
    print("Referral-ready after zone moves:", R["referral_ready_after_zone_moves"])
    print(f"\nFull-gap cost after redeployment: Rs {R['full_gap_cost_cr']} crore a year")
    print("Table 3 - budget scenarios")
    for b in R["budget_scenarios"]:
        print(f"  {int(b['budget_share']*100)}%  Rs {b['budget_cr']} cr  new N/P/D {b['N_new_posts']}/{b['P_new_posts']}/{b['D_new_posts']}"
              f"  share after D/N/P {b['D_share_after']}/{b['N_share_after']}/{b['P_share_after']}")
    print("Figure 4 - facilities functional by floor (25% budget):",
          {p["floor"]: (int(p["functional"]) if p["status"] == "Optimal" else "not affordable") for p in R["frontier_25pct_budget_target50"]})
    print("Table 4 - year each cadre reaches the norm:", R["year_norm_reached"])


if __name__ == "__main__":
    main(sens="--sens" in sys.argv)
