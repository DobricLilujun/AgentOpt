"""
CEMA analysis: compare the constraint-evolution result against the
no-constraint baseline and against the strategy-evolution system, and list
the "interesting" constraints the evolution discovered.

This is a NEW exploration module. It imports only from explore/ and reads
results; it does not modify the existing agents/.
"""
from __future__ import annotations
import sys, json
import numpy as np
sys.path.insert(0, ".")
import pandas as pd
from explore.constraint_evaluator import (evaluate_constraint, interestingness)
from explore.constraint_families import all_template_constraints


def main():
    tasks = pd.read_csv("results/real_tasks.csv")
    H = int(tasks["t_n"].max()) + 3

    # --- baseline (no constraints) ---
    base = evaluate_constraint(tasks, [], H, time_limit=10)
    print("=== BASELINE (no constraints) ===")
    print(f"  clusters={base['n_clusters']}  cost={base['cost']}  "
          f"cost_red={base['cost_reduction']*100:.1f}%  feasible={base['feasible']}")

    # --- each single constraint vs baseline ---
    print("\n=== SINGLE constraints vs baseline (Δcost<0 = helps the goal) ===")
    findings = []
    for c in all_template_constraints():
        m = evaluate_constraint(tasks, [c], H, time_limit=8)
        it = interestingness(base, m)
        dcost = m["cost"] - base["cost"] if m["feasible"] else None
        dcost_s = f"{dcost:+.0f}" if dcost is not None else " inf"
        print(f"  {c.name:18} feasible={str(m['feasible']):5} "
              f"cost={m['cost']:.0f}  Δcost={dcost_s:>8}  "
              f"interesting={it['interesting']} ({it['type']})")
        findings.append({"constraint": c.name, "feasible": m["feasible"],
                         "cost": m["cost"], "delta_cost": dcost,
                         "interesting": it})

    # --- CEMA best (from the run log / saved json) ---
    try:
        cea = json.load(open("explore/cea_result.json"))
        print("\n=== CEMA best constraint set ===")
        print(f"  set: {cea['best_set']}")
        print(f"  metrics: {cea['best_metrics']}")
        d = cea["best_metrics"]["cost"] - base["cost"]
        print(f"  Δcost vs baseline: {d:+.1f}  "
              f"({'helps' if d<0 else 'neutral/hurts'})")
    except FileNotFoundError:
        print("\n(no CEMA result json yet)")

    # --- summary of interesting constraints ---
    print("\n=== INTERESTING constraints discovered ===")
    interesting = [f for f in findings if f["interesting"]]
    if not interesting:
        print("  (none changed the outcome on this instance -- all neutral)")
    for f in interesting:
        print(f"  - {f['constraint']}: Δcost={f['delta_cost']} "
              f"type={f['interesting']['type']}")
    infeasible = [f for f in findings if not f["feasible"]]
    print("\n=== INFEASIBLE constraints (a finding in itself) ===")
    for f in infeasible:
        print(f"  - {f['constraint']}: cannot be satisfied -> infeasible")

    # sanitise numpy types for JSON
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    findings_clean = _clean(findings)
    base_clean = _clean(base)

    json.dump({"baseline": base_clean, "findings": findings_clean,
              "interesting": [f["constraint"] for f in interesting],
              "infeasible": [f["constraint"] for f in infeasible],
              "interesting_type": {f["constraint"]: f["interesting"]["type"]
                                   for f in findings}},
             open("explore/cea_analysis.json", "w"), indent=2)
    print("\nsaved explore/cea_analysis.json")


if __name__ == "__main__":
    main()