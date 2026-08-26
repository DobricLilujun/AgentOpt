"""
NEW SOLVER: OR-Tools CP-SAT (constraint programming).

The current solver is PuLP -> CBC (branch-and-bound MILP). This module
implements the SAME task-grouping model with a DIFFERENT solver engine:
Google OR-Tools CP-SAT (constraint propagation + local search) -- a
fundamentally different algorithm from branch-and-bound.

Both are compared on identical data to answer:
  "Can a new solver do the same thing, and how does it compare?"

Model (identical to the CBC ILP):
  minimise   sum_{d} y_d                      (number of clusters)
  subject to x_i in [day_a_i, day_b_i]        (time window)
             sum_i [x_i = d] <= C_max   /d    (per-day capacity)
             y_d = 1  iff  {i : x_i = d} non-empty
"""
from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model
from agents.grouping import GroupingAgent


def _bounds(tasks: pd.DataFrame, H: int) -> pd.DataFrame:
    return GroupingAgent()._bounds(tasks, H)


def solve_cpsat(tasks: pd.DataFrame, C_max: int, H: int,
                time_limit: float = 30.0) -> dict:
    """Solve the task-grouping model with OR-Tools CP-SAT (min clusters)."""
    df = _bounds(tasks, H)
    df = df.dropna(subset=["a_n", "b_n"])
    df = df[(df["a_n"] >= 0) & (df["b_n"] <= H)].reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {"status": "optimal", "n_clust": 0, "tasks": 0,
                "clusters": {d: [] for d in range(H + 1)},
                "clusters_occupied": [], "cost": 0}

    t = df["tid"].values
    day_a, day_b = df["a_n"].values, df["b_n"].values
    H_i = int(H)
    cap = int(C_max)

    model = cp_model.CpModel()
    # x[i] = cluster (day) assigned to task i
    x = [model.NewIntVar(0, H_i, f"x_{i}") for i in range(n)]
    # per-day membership z[d][i] = (x[i] == d)
    z = [[model.NewBoolVar(f"z_{d}_{i}") for i in range(n)] for d in range(H_i + 1)]
    y = [model.NewBoolVar(f"y_{d}") for d in range(H_i + 1)]

    for i in range(n):
        model.Add(x[i] >= day_a[i])
        model.Add(x[i] <= day_b[i])
    for d in range(H_i + 1):
        for i in range(n):
            model.Add(x[i] == d).OnlyEnforceIf(z[d][i])
            model.Add(x[i] != d).OnlyEnforceIf(z[d][i].Not())
        # capacity
        model.Add(sum(z[d]) <= cap)
        # y[d] = (sum z[d] >= 1)
        model.Add(sum(z[d]) >= 1).OnlyEnforceIf(y[d])
        model.Add(sum(z[d]) == 0).OnlyEnforceIf(y[d].Not())

    model.Minimize(sum(y))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_search_workers = 8
    status = sol.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        clusters = {d: [] for d in range(H_i + 1)}
        for i in range(n):
            d = sol.Value(x[i])
            clusters[d].append(int(t[i]))
        occ = [d for d in range(H_i + 1) if clusters[d]]
        return {"status": "optimal", "n_clust": len(occ), "tasks": n,
                "method": "CP-SAT",
                "assignment": {int(t[i]): sol.Value(x[i]) for i in range(n)},
                "active_days": occ,
                "clusters": {d: clusters[d] for d in occ},
                "clusters_occupied": occ, "cost": len(occ) * 10 + n}
    return {"status": "timeout/infeasible", "n_clust": n, "tasks": n,
            "method": "CP-SAT",
            "assignment": {int(t[i]): i for i in range(n)},
            "active_days": [i for i in range(n)],
            "clusters": {i: [int(t[i])] for i in range(n)},
            "clusters_occupied": [i for i in range(n)], "cost": n * 10 + n}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from agents import GroupingAgent, EvaluationAgent
    tasks = pd.read_csv("results/real_tasks.csv")
    print(f"real tasks: {len(tasks)} (t_n {tasks['t_n'].min():.1f}..{tasks['t_n'].max():.1f})")
    H = int(tasks["t_n"].max()) + 3

    # --- current solver: PuLP / CBC ---
    g = GroupingAgent(method="ilp", advance_prefer=0.0, cost_per_gr=10, time_limit=30)
    r_cbc = g.group(tasks, C_max=5, H=H)
    m_cbc = EvaluationAgent().evaluate(tasks, r_cbc, H=H)
    print(f"\n[CBC / PuLP ILP]  clusters={m_cbc['n_clusters']:3}  cost={m_cbc['cost']:5}  "
          f"cost_red={m_cbc['cost_reduction']*100:5.1f}%  leakage={m_cbc['leakage_kg']:.5f}kg  "
          f"viol={m_cbc['n_violations']}")

    # --- NEW solver: OR-Tools CP-SAT ---
    r_cs = solve_cpsat(tasks, C_max=5, H=H, time_limit=30)
    m_cs = EvaluationAgent().evaluate(tasks, r_cs, H=H)
    print(f"[CP-SAT/OR-Tools] clusters={m_cs['n_clusters']:3}  cost={m_cs['cost']:5}  "
          f"cost_red={m_cs['cost_reduction']*100:5.1f}%  leakage={m_cs['leakage_kg']:.5f}kg  "
          f"viol={m_cs['n_violations']}")

    # --- summary ---
    print("\n=== SOLVER COMPARISON (same tasks, same C_max=5, same model) ===")
    print(f"  CBC / PuLP  : {m_cbc['n_clusters']} clusters, cost={m_cbc['cost']}, "
          f"leakage={m_cbc['leakage_kg']:.5f}kg, violations={m_cbc['n_violations']}")
    print(f"  CP-SAT/OR-Tools: {m_cs['n_clusters']} clusters, cost={m_cs['cost']}, "
          f"leakage={m_cs['leakage_kg']:.5f}kg, violations={m_cs['n_violations']}")
    import json
    json.dump({"cbc": m_cbc, "cpsat": m_cs}, open("results/new_solver.json", "w"), indent=2)
    print("\nsaved results/new_solver.json")