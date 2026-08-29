"""
GroupingAgent -- Stage 2 of the maintenance pipeline.

Turns a set of pressure-service tasks into a grouped schedule (which tasks run
on which day) by minimising the number of deployment clusters, subject to
operational constraints. Two strategies, selectable per generation:

  - "ilp":    exact Integer Linear Programming (the core model,
              constraints 1-10). Objective: minimise # activated clusters.
  - "greedy": fast sliding-window heuristic.

A strategy can also add a small "advance-preference" penalty that biases tasks
toward being brought FORWARD (earlier) rather than delayed -- this addresses
the open point that the model has no advance/delay preference, and is one of
the knobs the EvolutionMaster evolves.

Implementation notes
---------------------
For speed in the multi-agent search, variables are built only over the feasible
[a_n, b_n] window of each task (never the whole horizon), and the CBC solver
runs with a time limit so a single generation never hangs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

P_NOM = 3.5
P_SERV = 3.2
P_CRIT = 3.0
SAFETY_MARGIN = 2  # conservative days before reaching the critical level P_CRIT


class GroupingAgent:
    def __init__(self, method: str = "ilp",
                 advance_prefer: float = 0.0,
                 window_len: int = 10,
                 advance_limit: int = 3,
                 cost_per_service: int = 10,
                 time_limit: int = 20):
        self.method = method
        self.advance_prefer = advance_prefer  # extra cost per day a task is advanced
        self.window_len = window_len
        self.advance_limit = advance_limit
        self.cost_per_service = cost_per_service
        self.time_limit = time_limit

    # ---- bounds (constraints 5-8) ---------------------------------
    def _bounds(self, tasks: pd.DataFrame, H: int) -> pd.DataFrame:
        df = tasks.copy().reset_index().rename(columns={"index": "orig_idx"})
        A = self.advance_limit
        df["a_n"] = df["t_n"].clip(lower=0).apply(lambda x: max(0, x - A))
        days_to_critical = (P_SERV - P_CRIT) / df["alpha"].clip(lower=1e-6)
        df["b_n"] = np.minimum(H, df["t_n"] + days_to_critical - SAFETY_MARGIN).round().astype(int)
        # cap the window to window_len days (a task may be delayed at most
        # window_len days beyond its trigger) so the ILP stays tractable at
        # large scale and no service is delayed unboundedly.
        df["b_n"] = np.maximum(df["b_n"], df["a_n"])
        if self.window_len:
            df["b_n"] = np.minimum(df["b_n"], df["a_n"] + self.window_len)
        df.loc[df["b_n"] < df["a_n"], "b_n"] = df.loc[df["b_n"] < df["a_n"], "a_n"]
        # never let a window extend beyond the planning horizon
        df["b_n"] = np.minimum(df["b_n"], H)
        return df

    # ---- ILP solver (core model + optional advance-preference penalty) --
    def _solve_ilp(self, tasks: pd.DataFrame, C_max: int, H: int) -> dict:
        df = self._bounds(tasks, H)
        prob = pulp.LpProblem("grouping", pulp.LpMinimize)

        # Build vars only over each task's feasible window.
        x = {}
        by_day: dict[int, list] = {}  # day -> list of (tid, var)
        by_task: dict[int, list] = {}  # tid -> list of vars
        for r in df.itertuples():
            vars_r = []
            for d in range(int(r.a_n), int(r.b_n) + 1):
                v = pulp.LpVariable(f"x_{r.tid}_{d}", cat="Binary")
                x[(r.tid, d)] = v
                by_day.setdefault(d, []).append((r.tid, v))
                vars_r.append(v)
            by_task[r.tid] = vars_r
        y = {d: pulp.LpVariable(f"y_{d}", cat="Binary") for d in range(H + 1)}

        # (1) objective: minimise activated clusters (+ optional advance penalty)
        obj = pulp.lpSum(y[d] for d in range(H + 1))
        if self.advance_prefer:
            obj = obj + self.advance_prefer * pulp.lpSum(
                (int(r.t_n) - d) * x[(r.tid, d)]
                for r in df.itertuples()
                for d in range(int(r.a_n), int(r.b_n) + 1))
        prob += obj

        # (3) assign each task once
        for tid, vars_r in by_task.items():
            prob += pulp.lpSum(vars_r) == 1
        # (2) theta <= cluster
        for d, lst in by_day.items():
            for (tid, v) in lst:
                prob += v <= y[d]
        # (4) capacity per cluster/day
        for d, lst in by_day.items():
            if len(lst) > C_max:
                prob += pulp.lpSum(v for (_, v) in lst) <= C_max

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
        assignment = {}
        for tid, vars_r in by_task.items():
            for d in range(0, H + 1):
                v = x.get((tid, d))
                if v is not None and v.value() and v.value() > 0.5:
                    assignment[tid] = d
        active_days = [d for d in range(H + 1) if y[d].value() and y[d].value() > 0.5]
        return {"method": "ILP", "status": pulp.LpStatus[prob.status],
                "assignment": assignment, "active_days": active_days}

    # ---- greedy sliding-window heuristic -------------------------------
    def _solve_greedy(self, tasks: pd.DataFrame, C_max: int, H: int) -> dict:
        df = self._bounds(tasks, H).sort_values("t_n").reset_index(drop=True)
        assignment, clusters = {}, {}

        def _place(tid, lo, hi):
            # scan candidate days in the window; attach to a day with room.
            order = list(range(lo, hi + 1))
            order = sorted(order, reverse=(self.advance_prefer > 0))
            for day in order:
                if len(clusters.get(day, [])) < C_max:
                    clusters.setdefault(day, []).append(tid)
                    return day
            # no day in the window has capacity -> open a new one at the
            # preferred edge (still in-window, so it stays feasible)
            new_day = hi if self.advance_prefer > 0 else lo
            clusters.setdefault(new_day, []).append(tid)
            return new_day

        for _, row in df.iterrows():
            tid = row.tid
            day = _place(tid, int(row.a_n), int(row.b_n))
            assignment[tid] = day
        active_days = [d for d, t in clusters.items() if t]
        return {"method": "Greedy", "status": "OK",
                "assignment": assignment, "active_days": active_days}

    def group(self, tasks: pd.DataFrame, C_max: int = 5, H: int = 30) -> dict:
        if self.method == "greedy":
            return self._solve_greedy(tasks, C_max, H)
        return self._solve_ilp(tasks, C_max, H)