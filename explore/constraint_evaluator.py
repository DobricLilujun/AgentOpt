"""
ConstraintEvaluator -- the agent that ADDS a constraint to the problem and
measures whether it helps the fixed GOAL, is feasible, and is "interesting".

GOAL (fixed, from user choice A): minimise total cost, with 0 reliability
violations.

The base problem (the "no-constraint baseline") is the task-grouping model:
  min  clusters
  s.t.  each task once;
       time window  x_i in [t_n - A_base, t_n + D_base];
       per-day capacity  sum_i [x_i = d] <= C_max.
A candidate CONSTRAINT is added ON TOP of this base, and the evaluator
reports feasibility, clusters, cost and an "interestingness" score relative
to the baseline.

There is no fluid-leakage axis: in this hydraulic station group a pressure
problem is only a slow pressure drop, not a leak of working fluid, and a
repair returns the unit to its nominal pressure and continues. The cost is
therefore deployment (transport + personnel) plus a reliability term.

This is a NEW module; it does not import or modify the existing agents/.
It reuses OR-Tools CP-SAT (the new solver already installed).
"""
from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model


# base window for the real data (advance 3 days, delay up to 15 days to threshold)
A_BASE = 3
D_BASE = 15
C_MAX = 5
C_DEP = 10          # deployment cost per cluster
REL_COST = 500.0    # cost per reliability violation


def _base_model(tasks: pd.DataFrame, H: int):
    """Build the base grouping CP-SAT model (no extra constraints)."""
    df = tasks.dropna(subset=["t_n"]).reset_index(drop=True)
    n = len(df)
    t_n = df["t_n"].values
    model = cp_model.CpModel()
    x = [model.NewIntVar(0, H, f"x_{i}") for i in range(n)]
    y = [model.NewBoolVar(f"y_{d}") for d in range(H + 1)]
    z = [[model.NewBoolVar(f"z_{d}_{i}") for i in range(n)] for d in range(H + 1)]
    for i in range(n):
        model.Add(x[i] >= t_n[i] - A_BASE)          # earliest (advance limit)
        model.Add(x[i] <= t_n[i] + D_BASE)          # latest (to threshold)
    for d in range(H + 1):
        for i in range(n):
            model.Add(x[i] == d).OnlyEnforceIf(z[d][i])
            model.Add(x[i] != d).OnlyEnforceIf(z[d][i].Not())
        model.Add(sum(z[d]) <= C_MAX)              # capacity
        model.Add(sum(z[d]) >= 1).OnlyEnforceIf(y[d])
        model.Add(sum(z[d]) == 0).OnlyEnforceIf(y[d].Not())
    # objective: minimise clusters, with a composite-cost tie-breaker (a small
    # penalty for advancing tasks early) so the optimum is UNIQUE /
    # reproducible instead of CP-SAT picking any of the many min-cluster
    # solutions.
    advance = [model.NewIntVar(-A_BASE, H, f"adv_{i}") for i in range(n)]
    for i in range(n):
        model.Add(advance[i] == x[i] - t_n[i])
    pos = [model.NewIntVar(0, H, f"pos_{i}") for i in range(n)]
    for i in range(n):
        model.AddMaxEquality(pos[i], [advance[i], 0])
    model.Minimize(sum(10 * y[d] for d in range(H + 1)) +
                   sum(20 * pos[i] for i in range(n)))
    return model, x, y, z, df, t_n


def evaluate_constraint(tasks: pd.DataFrame, constraints: list,
                        H: int, time_limit: float = 20.0) -> dict:
    """Solve the base problem + the given (active) constraints."""
    model, x, y, z, df, t_n = _base_model(tasks, H)
    # apply the active constraints
    for c in constraints:
        if getattr(c, "active", True):
            c.apply(model, df, x, H)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_search_workers = 1   # deterministic, reproducible
    status = sol.Solve(model)
    n = len(df)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        clusters = {d: [] for d in range(H + 1)}
        x_days = [0] * n
        for i in range(n):
            clusters[sol.Value(x[i])].append(i)
            x_days[i] = sol.Value(x[i])
        occ = [d for d in range(H + 1) if clusters[d]]
        deploy = len(occ) * C_DEP + n
        reliability = sum(max(t_n[i] - x_days[i], 0) for i in range(n))  # days past
        # composite total cost (the GOAL): deployment + reliability
        total = deploy + REL_COST * reliability
        return {"status": "optimal", "feasible": True,
                "n_clusters": len(occ), "cost": total,
                "deploy_cost": deploy,
                "n_violations": 0,
                "tasks": n,
                "cost_reduction": 1 - deploy / (n * (C_DEP + 1))}
    return {"status": "infeasible", "feasible": False,
            "n_clusters": n, "cost": 1e8,
            "deploy_cost": n * C_DEP + n, "n_violations": 0,
            "tasks": n,
            "cost_reduction": 1 - (n * C_DEP + n) / (n * (C_DEP + 1))}


def interestingness(baseline: dict, candidate: dict) -> dict:
    """Score how 'interesting' a candidate constraint is, relative to the
    baseline. Interesting = helps the goal, or reveals a trade-off, or is
    infeasible (a finding in itself)."""
    if not candidate["feasible"]:
        # an infeasible constraint is itself a finding
        return {"interesting": True, "type": "infeasible",
                "delta_cost": None, "delta_clusters": None}
    d_cost = candidate["cost"] - baseline["cost"]
    d_cl = candidate["n_clusters"] - baseline["n_clusters"]
    if d_cost < 0:
        itype = "helps"            # improves the goal
    elif d_cost > 0:
        itype = "hurts"           # worsens the goal
    else:
        itype = "neutral"
    interesting = (d_cost != 0)   # anything that changes the outcome
    return {"interesting": interesting, "type": itype,
            "delta_cost": d_cost, "delta_clusters": d_cl}


def _summary(label: str, m: dict, base: dict = None) -> str:
    s = (f"  {label:34} feasible={str(m['feasible']):5}  "
         f"clusters={m['n_clusters']:2}  cost={m['cost']:4}  "
         f"cost_red={m['cost_reduction']*100:5.1f}%")
    if base is not None:
        d = m["cost"] - base["cost"]
        s += f"  (Δcost={d:+d})"
    return s


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from explore.constraint_families import all_template_constraints
    tasks = pd.read_csv("results/real_tasks.csv")
    H = int(tasks["t_n"].max()) + 3
    base = evaluate_constraint(tasks, [], H)
    print("BASELINE (no constraints)")
    print(_summary("no constraints", base, base))
    print("\n+ each single constraint")
    for c in all_template_constraints():
        m = evaluate_constraint(tasks, [c], H)
        print(_summary(c.name, m, base))