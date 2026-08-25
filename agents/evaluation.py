"""
EvaluationAgent -- scores a grouped schedule on the real objective of the
problem, and turns that score into a fitness for the EvolutionMaster.

The objective is a TRADE-OFF (as the paper emphasises) between:
  - cost        : deployment (transport + personnel) cost, ~ cost_per_gr * #clusters
  - leakage     : SF6 leaked before each GR; a task scheduled LATER leaks MORE
                  (pressure sits lower for longer), so grouping that DELAYS a
                  task increases leakage. This is the environmental axis.
  - reliability : hard safety -- a task scheduled past its latest feasible day
                  b_n lets the unit drop below the 3.0 bar threshold -> fault.
                  Any violation is a large penalty (near-disqualifying).

The composite fitness weights these three axes with learned weights, so the
EvolutionMaster can evolve the trade-off (e.g. lean toward cost, or toward
environmental impact) and discover strategies the paper's single-objective ILP
cannot reach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_MAX = 3.5
P_GR_TRIGGER = 3.2
P_CRITICAL = 3.0
LEAK_RATE_PER_BAR_DAY = 0.0005  # kg SF6 leaked per bar-day of undershoot (model constant)


def _feasibility(tasks: pd.DataFrame, assignment: dict, H: int,
                advance_limit: int = 3, safety_margin: int = 2) -> pd.DataFrame:
    """Compute a_n, b_n for each task (paper constraints 5-8)."""
    df = tasks.copy().reset_index().rename(columns={"index": "orig_idx"})
    df["a_n"] = df["t_n"].clip(lower=0).apply(lambda x: max(0, x - advance_limit))
    days_to_critical = (P_GR_TRIGGER - P_CRITICAL) / df["alpha"].clip(lower=1e-6)
    df["b_n"] = np.minimum(H, df["t_n"] + days_to_critical - safety_margin).round().astype(int)
    df.loc[df["b_n"] < df["a_n"], "b_n"] = df.loc[df["b_n"] < df["a_n"], "a_n"]
    return df


class EvaluationAgent:
    def __init__(self,
                 w_cost: float = 1.0,
                 w_leak: float = 1.0,
                 w_reliability: float = 5.0,
                 cost_per_gr: int = 10,
                 advance_limit: int = 3,
                 safety_margin: int = 2):
        self.w_cost = w_cost
        self.w_leak = w_leak
        self.w_reliability = w_reliability
        self.cost_per_gr = cost_per_gr
        self.advance_limit = advance_limit
        self.safety_margin = safety_margin

    def evaluate(self, tasks: pd.DataFrame, result: dict,
                 H: int = 30) -> dict:
        """Score a grouping result; return metrics + a composite fitness."""
        df = _feasibility(tasks, result["assignment"], H,
                          self.advance_limit, self.safety_margin)
        df["scheduled_day"] = df["tid"].map(result["assignment"])

        n_tasks = len(df)
        n_clusters = len({d for d in result["active_days"] if d is not None})

        # cost = cost_per_gr * clusters
        cost = n_clusters * self.cost_per_gr
        cost_base = n_tasks * self.cost_per_gr

        # leakage: for each task, leak ~ alpha * (days below trigger before service)
        # a task scheduled at `scheduled_day` leaks alpha*(scheduled_day - t_n) extra
        # relative to its original trigger; advancing reduces it, delaying increases it.
        df["leak"] = df["alpha"] * (df["scheduled_day"] - df["t_n"]).clip(lower=0) * LEAK_RATE_PER_BAR_DAY
        leakage = float(df["leak"].sum())

        # reliability: a violation is a task scheduled past its latest feasible day
        df["violation"] = df["scheduled_day"] > df["b_n"] + 1e-6
        n_violations = int(df["violation"].sum())
        reliability = 1.0 - n_violations / max(n_tasks, 1)
        reliability_penalty = float(n_violations)  # hard penalty

        # advance/delay profile (the paper's "advance vs delay" diagnostic)
        df["shift"] = df["scheduled_day"] - df["t_n"]
        advance = float((df["shift"] < 0).sum())
        delay = float((df["shift"] > 0).sum())

        # composite fitness (lower is better); reliability dominates
        fitness = (self.w_cost * cost
                   + self.w_leak * leakage * 1000.0
                   + self.w_reliability * reliability_penalty * 100.0)
        return {
            "method": result.get("method", "?"),
            "n_clusters": n_clusters,
            "n_tasks": n_tasks,
            "cost": cost,
            "cost_base": cost_base,
            "cost_reduction": 1 - cost / cost_base if cost_base else 0.0,
            "leakage_kg": leakage,
            "n_violations": n_violations,
            "reliability": reliability,
            "advance": advance,
            "delay": delay,
            "fitness": fitness,
        }