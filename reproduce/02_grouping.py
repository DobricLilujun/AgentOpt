"""
Stage 2 - Task grouping optimization.

Two solvers:
  (A) Exact Integer Linear Programming  -- the exact grouping model
      (constraints 1-10). Objective (1): minimise the number of activated
      maintenance clusters (== deployments). This is the core model.
  (B) Greedy sliding-window heuristic  -- a fast baseline and a component
      inside the multi-agent system.

Window / feasibility (constraints 5-8):
    a_n = max(0, t_n - A)                       (earliest a task may move to)
    b_n = min(H, t_n + (P_SERV - P_CRIT)/alpha - SAFETY_MARGIN)
      where P_SERV and P_CRIT are the service-trigger and critical (safety-floor)
      levels, so the (P_SERV - P_CRIT)/alpha term is the "days until the unit
      reaches the critical level minus a safety margin".

Capacity (constraint 4): at most C_MAX tasks per cluster (day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

try:
    from . import degrade as D  # when run as a package (from project root)
except ImportError:  # pragma: no cover
    import degrade as D  # when run as a script


def window_bounds(df: pd.DataFrame, A: int = D.A, H: int = D.T) -> pd.DataFrame:
    """Compute a_n (earliest) and b_n (latest) shift bounds for each task."""
    df = df.copy()
    df = df.reset_index().rename(columns={"index": "orig_idx"})
    df["a_n"] = df["t_n"].clip(lower=0).apply(lambda x: max(0, x - A))
    days_to_critical = (D.P_SERV - D.P_CRIT) / df["alpha"]
    df["b_n"] = np.minimum(H, df["t_n"] + days_to_critical - D.SAFETY_MARGIN).round().astype(int)
    # never allow b < a
    df.loc[df["b_n"] < df["a_n"], "b_n"] = df.loc[df["b_n"] < df["a_n"], "a_n"]
    return df


def solve_ilp(df: pd.DataFrame,
              C_max: int = D.C_MAX,
              H: int = D.T) -> dict:
    """Solve the exact grouping ILP (constraints 1-10) with PuLP.

    Returns a dict with the assignment, cluster days, objective and metrics.
    """
    df = window_bounds(df, H=H)
    prob = pulp.LpProblem("service_task_grouping", pulp.LpMinimize)

    # decision vars: x[(tid, d)] binary, y[d] cluster-active
    x = {}
    for r in df.itertuples():
        for d in range(int(r.a_n), int(r.b_n) + 1):
            x[(r.tid, d)] = pulp.LpVariable(f"x_{r.tid}_{d}", cat="Binary")
    y = {d: pulp.LpVariable(f"y_{d}", cat="Binary") for d in range(H + 1)}

    # (1) objective: minimise activated clusters
    prob += pulp.lpSum(y[d] for d in range(H + 1))

    # (3) each task assigned exactly once
    for r in df.itertuples():
        prob += (pulp.lpSum(x[(r.tid, d)] for d in range(int(r.a_n), int(r.b_n) + 1)) == 1)

    # (2) theta_nt <= cluster_t
    for r in df.itertuples():
        for d in range(int(r.a_n), int(r.b_n) + 1):
            prob += x[(r.tid, d)] <= y[d]

    # (4) capacity per cluster/day
    for d in range(H + 1):
        prob += pulp.lpSum(x[(n, d)] for (n, dd) in x if dd == d) <= C_max

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    assignment = {}
    for r in df.itertuples():
        for d in range(int(r.a_n), int(r.b_n) + 1):
            if x[(r.tid, d)].value() and x[(r.tid, d)].value() > 0.5:
                assignment[r.tid] = d
    active_days = [d for d in range(H + 1) if y[d].value() and y[d].value() > 0.5]

    n_clusters = len(active_days)
    n_tasks = len(df)
    cost = n_clusters * D.COST_PER_SERVICE
    cost_base = n_tasks * D.COST_PER_SERVICE
    return {
        "method": "ILP",
        "status": pulp.LpStatus[prob.status],
        "assignment": assignment,
        "active_days": active_days,
        "n_clusters": n_clusters,
        "n_tasks": n_tasks,
        "cost": cost,
        "cost_base": cost_base,
        "reduction": (1 - cost / cost_base) if cost_base else 0.0,
    }


def solve_greedy(df: pd.DataFrame,
                 C_max: int = D.C_MAX,
                 H: int = D.T) -> dict:
    """Greedy sliding-window heuristic.

    Sort tasks by their original time; attach each task to an existing
    in-window cluster if capacity allows, else open a new cluster at the
    earliest feasible day (a slight preference for advancing).
    """
    df = window_bounds(df, H=H)
    df = df.sort_values("t_n").reset_index(drop=True)

    assignment = {}
    clusters: dict[int, list] = {}

    for _, row in df.iterrows():
        tid = row.tid
        lo, hi = int(row.a_n), int(row.b_n)
        placed = False
        for day in sorted(clusters.keys()):
            if lo <= day <= hi and len(clusters[day]) < C_max:
                clusters[day].append(tid)
                assignment[tid] = day
                placed = True
                break
        if not placed:
            new_day = lo
            clusters.setdefault(new_day, []).append(tid)
            assignment[tid] = new_day

    active_days = [d for d, t in clusters.items() if t]
    n_clusters = len(active_days)
    n_tasks = len(df)
    cost = n_clusters * D.COST_PER_SERVICE
    cost_base = n_tasks * D.COST_PER_SERVICE
    return {
        "method": "Greedy",
        "assignment": assignment,
        "active_days": active_days,
        "n_clusters": n_clusters,
        "n_tasks": n_tasks,
        "cost": cost,
        "cost_base": cost_base,
        "reduction": (1 - cost / cost_base) if cost_base else 0.0,
    }


if __name__ == "__main__":
    from degrade import generate_maintenance_plan
    plan = generate_maintenance_plan(seed=0)
    print(f"Stage1 tasks: {len(plan)}  (count is seed-dependent)")
    ilp = solve_ilp(plan)
    gr = solve_greedy(plan)
    print("ILP   :", ilp["status"], f"clusters={ilp['n_clusters']}",
          f"cost={ilp['cost']}/{ilp['cost_base']}  reduction={ilp['reduction']:.1%}")
    print("Greedy:", f"clusters={gr['n_clusters']}",
          f"cost={gr['cost']}/{gr['cost_base']}  reduction={gr['reduction']:.1%}")