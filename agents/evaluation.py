"""
EvaluationAgent -- scores a grouped schedule on the real objective of the
problem, and turns that score into a fitness for the EvolutionMaster.

The objective is a TRADE-OFF between two axes:
  - cost        : deployment (transport + personnel) cost, ~ cost_per_service * #clusters.
                  Fewer clusters (deployments) is cheaper.
  - reliability : hard safety -- a task scheduled past its latest feasible day
                  b_n lets the unit drop below the critical level P_CRIT -> fault.
                  Any violation is a large penalty (near-disqualifying).

(There is no fluid-leakage axis: in this hydraulic station group a pressure
problem is only a slow pressure DROP, not a real leak of working fluid, and a
repair simply returns the unit to its nominal pressure P_NOM and continues.
There is therefore no mass of fluid that "leaks" to cost.)

The composite fitness minimises deployment cost, with reliability enforced as a
HARD constraint (a violation is a large, near-disqualifying penalty whose
severity the EvolutionMaster can evolve via w_reliability). This lets the
system discover scheduling strategies that cut cost while guaranteeing the
safety constraint -- something a single-objective ILP (which fixes the safety
margin) cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_NOM = 3.5
P_SERV = 3.2
P_CRIT = 3.0


def _feasibility(tasks: pd.DataFrame, assignment: dict, H: int,
                advance_limit: int = 3, safety_margin: int = 2) -> pd.DataFrame:
    """Compute a_n, b_n for each task (constraints 5-8)."""
    df = tasks.copy().reset_index().rename(columns={"index": "orig_idx"})
    df["a_n"] = df["t_n"].clip(lower=0).apply(lambda x: max(0, x - advance_limit))
    days_to_critical = (P_SERV - P_CRIT) / df["alpha"].clip(lower=1e-6)
    df["b_n"] = np.minimum(H, df["t_n"] + days_to_critical - safety_margin).round().astype(int)
    df.loc[df["b_n"] < df["a_n"], "b_n"] = df.loc[df["b_n"] < df["a_n"], "a_n"]
    return df


class EvaluationAgent:
    def __init__(self,
                 w_reliability: float = 5.0,
                 cost_per_service: int = 10,
                 advance_limit: int = 3,
                 safety_margin: int = 2):
        self.w_reliability = w_reliability
        self.cost_per_service = cost_per_service
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

        # cost = cost_per_service * clusters
        cost = n_clusters * self.cost_per_service
        cost_base = n_tasks * self.cost_per_service

        # reliability: a violation is a task scheduled past its latest feasible day
        df["violation"] = df["scheduled_day"] > df["b_n"] + 1e-6
        n_violations = int(df["violation"].sum())
        reliability = 1.0 - n_violations / max(n_tasks, 1)
        reliability_penalty = float(n_violations)  # hard penalty

        # advance/delay profile (the "advance vs delay" diagnostic, informational)
        df["shift"] = df["scheduled_day"] - df["t_n"]
        advance = float((df["shift"] < 0).sum())
        delay = float((df["shift"] > 0).sum())

        # fitness (lower is better).
        # Reliability is a HARD safety requirement, not a tunable trade-off:
        # a schedule with any violation is penalised heavily (near-disqualifying)
        # so the optimiser never accepts a cheaper-but-unsafe schedule.
        # With 0 violations (the normal case) the fitness is just the deployment
        # cost, so minimising clusters == minimising cost.
        fitness = (cost
                   + self.w_reliability * reliability_penalty * 1e5)
        return {
            "method": result.get("method", "?"),
            "n_clusters": n_clusters,
            "n_tasks": n_tasks,
            "cost": cost,
            "cost_base": cost_base,
            "cost_reduction": 1 - cost / cost_base if cost_base else 0.0,
            "n_violations": n_violations,
            "reliability": reliability,
            "advance": advance,
            "delay": delay,
            "fitness": fitness,
        }