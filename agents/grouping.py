"""
GroupingAgent -- Stage 2 of the maintenance pipeline.

Turns a set of pressure-service tasks into a grouped schedule (which tasks run
on which day) by minimising the number of deployment clusters, subject to
operational constraints. Two strategies, selectable per generation:

  - "ilp":    exact Integer Linear Programming (the core model).
  - "greedy": fast sliding-window heuristic.

Constraints modelled
---------------------
  A "cluster" is a group of tasks of a SINGLE repair type done in one
  deployment.  Because the two repair types (A: repressurisation, B:
  seal-replacement) require different technician specialities, a cluster is
  identified by (day, repair_type): a day may carry at most one A-cluster and
  one B-cluster.

  (a) each task is scheduled exactly once, inside its feasible window;
  (b) a (day, type) cluster is active only if it serves a task;
  (c) CAPACITY: at most C_max tasks of one type may be done on one day;
  (d) CROSS-TYPE is enforced structurally by keying clusters on (day, type):
      two tasks of different types can never share a cluster;
  (e) DEFERRED: a task whose log says "本周不建议维修该 unit" cannot be
      scheduled in its trigger week and is pushed to the following week.
  (f) a schedule with any reliability violation is penalised (see
      EvaluationAgent).

The number of clusters is therefore the deployment cost.  Because a day with
both an A- and a B-task needs two clusters, the cross-type constraint makes the
problem strictly harder than an untyped one -- which is exactly the point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

P_NOM = 3.5
P_SERV = 3.2
P_CRIT = 3.0
SAFETY_MARGIN = 2  # conservative days before reaching the critical level P_CRIT
WEEK = 7
TYPES = ("A", "B")


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

    # ---- bounds (feasibility window of each task) -----------------------
    def _bounds(self, tasks: pd.DataFrame, H: int) -> pd.DataFrame:
        df = tasks.copy().reset_index().rename(columns={"index": "orig_idx"})
        A = self.advance_limit
        df["a_n"] = df["t_n"].clip(lower=0).apply(lambda x: max(0, x - A))
        days_to_critical = (P_SERV - P_CRIT) / df["alpha"].clip(lower=1e-6)
        df["b_n"] = np.minimum(H, df["t_n"] + days_to_critical - SAFETY_MARGIN).round().astype(int)
        df["b_n"] = np.maximum(df["b_n"], df["a_n"])
        # (e) DEFERRED: a "本周不建议维修" task cannot be scheduled in its
        #     trigger week; shift its window to start at the following week.
        deferred = df.get("deferred")
        if deferred is not None and bool(deferred.any()):
            next_week = 7 * (df.loc[deferred, "t_n"] // WEEK + 1).astype(int)
            df.loc[deferred, "a_n"] = np.maximum(df.loc[deferred, "a_n"],
                                                next_week[deferred].values)
        if self.window_len:
            df["b_n"] = np.minimum(df["b_n"], df["a_n"] + self.window_len)
        df.loc[df["b_n"] < df["a_n"], "b_n"] = df.loc[df["b_n"] < df["a_n"], "a_n"]
        df["b_n"] = np.minimum(df["b_n"], H)
        return df

    # ---- ILP solver: clusters keyed on (day, repair_type) ---------------
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

        # a (day, type) cluster is active only if it serves a task.
        y = {}
        for d in range(H + 1):
            for rt in TYPES:
                y[(d, rt)] = pulp.LpVariable(f"y_{d}_{rt}", cat="Binary")

        # (1) objective: minimise the number of active (day, type) clusters
        #     (+ optional advance-preference penalty).
        obj = pulp.lpSum(y[(d, rt)] for d in range(H + 1) for rt in TYPES)
        if self.advance_prefer:
            obj = obj + self.advance_prefer * pulp.lpSum(
                (int(r.t_n) - d) * x[(r.tid, d)]
                for r in df.itertuples()
                for d in range(int(r.a_n), int(r.b_n) + 1))
        prob += obj

        # (a) assign each task once
        for tid, vars_r in by_task.items():
            prob += pulp.lpSum(vars_r) == 1
        # (b) task on day d activates the (d, type) cluster of its own type
        for r in df.itertuples():
            rt = getattr(r, "repair_type", "A")
            for d in range(int(r.a_n), int(r.b_n) + 1):
                prob += x[(r.tid, d)] <= y[(d, rt)]
        # (c) CAPACITY: at most C_max tasks of one type on one day.
        for d, lst in by_day.items():
            for rt in TYPES:
                same = [v for (tid, v) in lst
                       if str(df.loc[tid, "repair_type"]) == rt]
                if len(same) > C_max:
                    prob += pulp.lpSum(same) <= C_max

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
        # If the ILP is infeasible (e.g. C_max too small to fit a peak day)
        # or could not be solved to completion in time, fall back to the
        # greedy heuristic so we always return a FULL, feasible schedule.
        # A "Not Solved" (timed-out) ILP may be a PARTIAL solution (only some
        # tasks assigned -> a tiny cluster count); keeping it would let the
        # optimiser "win" on an incomplete schedule, so we never accept it.
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            return self._solve_greedy(tasks, C_max, H)
        # guard: even an "Optimal" result must assign every task; if CBC
        # returned a degenerate optimum, fall back to the greedy heuristic.
        assigned = sum(1 for v in x.values() if v is not None and v.value() and v.value() > 0.5)
        if assigned < len(tasks):
            return self._solve_greedy(tasks, C_max, H)
        assignment = {}
        for tid, vars_r in by_task.items():
            for d in range(0, H + 1):
                v = x.get((tid, d))
                if v is not None and v.value() and v.value() > 0.5:
                    assignment[tid] = d
        active_days = []
        for d in range(H + 1):
            for rt in TYPES:
                if y[(d, rt)].value() and y[(d, rt)].value() > 0.5:
                    active_days.append((d, rt))
        # report as active (day,type) clusters but also the number of active
        # calendar days, for the cluster count used by the evaluation agent.
        return {"method": "ILP", "status": status,
                "assignment": assignment,
                "active_days": [d for d, rt in active_days],
                "n_clusters": len(active_days)}

    # ---- greedy sliding-window heuristic -------------------------------
    def _solve_greedy(self, tasks: pd.DataFrame, C_max: int, H: int) -> dict:
        df = self._bounds(tasks, H).sort_values("t_n").reset_index(drop=True)
        assignment, clusters = {}, {}

        def _place(tid, lo, hi, rtype):
            # scan candidate days in the window; attach to a day that has room
            # for this repair type (a day can hold one A-cluster + one B-cluster).
            order = list(range(lo, hi + 1))
            order = sorted(order, reverse=(self.advance_prefer > 0))
            for day in order:
                same = clusters.get((day, rtype), [])
                if len(same) < C_max:
                    clusters[(day, rtype)] = same + [tid]
                    return day
            # no same-type slot within the window for this task: open a NEW
            # cluster at the preferred edge (still in-window, so it stays
            # feasible) that is itself within C_max. NEVER overload a cluster
            # beyond C_max -- doing so produced a physically-impossible count
            # (e.g. 10 clusters for 378 tasks) that the optimiser could "win"
            # on. If the preferred-edge day is full, open a fresh day just
            # beyond it (kept inside the window's type cluster space).
            new_day = hi if self.advance_prefer > 0 else lo
            existing = clusters.get((new_day, rtype), [])
            if len(existing) < C_max:
                clusters[(new_day, rtype)] = existing + [tid]
                return new_day
            # the preferred edge is full; find any in-window day with room,
            # else open a fresh (day, type) cluster adjacent to the window.
            for day in range(lo, hi + 1):
                if len(clusters.get((day, rtype), [])) < C_max:
                    clusters[(day, rtype)] = clusters.get((day, rtype), []) + [tid]
                    return day
            fresh = hi + 1 if self.advance_prefer > 0 else lo - 1
            while 0 <= fresh <= H and len(clusters.get((fresh, rtype), [])) >= C_max:
                fresh += 1 if self.advance_prefer > 0 else -1
            clusters[(fresh, rtype)] = [tid]
            return fresh

        for _, row in df.iterrows():
            tid = row.tid
            rtype = getattr(row, "repair_type", "A")
            day = _place(tid, int(row.a_n), int(row.b_n), rtype)
            assignment[tid] = day
        active_days = [d for d, rt in clusters if clusters[(d, rt)]]
        return {"method": "Greedy", "status": "OK",
                "assignment": assignment, "active_days": active_days,
                "n_clusters": len(active_days)}

    def group(self, tasks: pd.DataFrame, C_max: int = 5, H: int = 30) -> dict:
        if self.method == "greedy":
            return self._solve_greedy(tasks, C_max, H)
        return self._solve_ilp(tasks, C_max, H)