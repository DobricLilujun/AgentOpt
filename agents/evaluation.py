"""
EvaluationAgent -- scores a grouped schedule on the real objective of the
problem, and turns that score into a fitness for the EvolutionMaster.

The objective is a TRADE-OFF between THREE axes:
  - cost        : deployment (transport + personnel) cost, ~ cost_per_service * #clusters.
                  Fewer clusters (deployments) is cheaper.
  - reliability : hard safety -- a task scheduled past its latest feasible day
                  b_n lets the unit drop below the critical level P_CRIT -> fault.
                  Any violation is a large penalty (disqualifying -> fitness = inf).
  - schedule-shift : moving a task away from its optimal (natural) time t_n has a
                  cost, asymmetric by direction:
                      * advancing a task (earlier than t_n) costs C_a per day-unit;
                      * delaying  a task (later  than t_n) costs C_d per day-unit,
                        with C_d > C_a because delaying a unit past its nominal
                        service time raises the probability it drops to a fault
                        before the repair (a failure-risk penalty).
                  A task left at t_n (not moved) incurs no shift penalty, so the
                  "optimal time" is the natural time.

The composite fitness minimises
      cost + lambda * (C_a * advance_days + C_d * delay_days)
with reliability enforced as a HARD constraint (any violation -> fitness = inf).
With zero violations the fitness has two live terms -- the deployment cost AND
the (asymmetric) schedule-shift penalty -- so the optimiser trades off
fewer-deployments against staying-close-to-optimal-time, instead of collapsing
to a single objective.
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
                 safety_margin: int = 2,
                 advance_cost: float = 0.05,
                 delay_cost: float = 0.25,
                 lambda_shift: float = 1.0):
        self.w_reliability = w_reliability
        self.cost_per_service = cost_per_service
        self.advance_limit = advance_limit
        self.safety_margin = safety_margin
        # asymmetric schedule-shift costs (per day-unit of shift), with
        # delay_cost > advance_cost because delaying a unit raises failure risk.
        self.advance_cost = advance_cost      # C_a
        self.delay_cost = delay_cost          # C_d  (> C_a)
        self.lambda_shift = lambda_shift      # weight of the shift term vs cost

    def evaluate(self, tasks: pd.DataFrame, result: dict,
                 H: int = 30) -> dict:
        """Score a grouping result; return metrics + a composite fitness."""
        df = _feasibility(tasks, result["assignment"], H,
                          self.advance_limit, self.safety_margin)
        df["scheduled_day"] = df["tid"].map(result["assignment"])

        n_tasks = len(df)
        # a cluster is a (day, repair_type) group; the grouping agent reports
        # this as n_clusters (a day with both an A- and a B-task is 2 clusters).
        n_clusters = result.get("n_clusters",
                                len({d for d in result["active_days"]
                                     if d is not None}))

        # cost = cost_per_service * clusters
        cost = n_clusters * self.cost_per_service
        cost_base = n_tasks * self.cost_per_service

        # reliability: a violation is a task scheduled past its latest feasible day,
        # OR a task that was never scheduled at all (dropped -> the unit is
        # never serviced -> it will certainly fall below P_CRIT). Counting
        # dropped tasks as violations is what keeps the optimiser from
        # "winning" by dropping tasks (which would yield a tiny,
        # physically-impossible cluster count).
        df["violation"] = df["scheduled_day"] > df["b_n"] + 1e-6
        df["unassigned"] = df["scheduled_day"].isna()
        df["violation"] = df["violation"] | df["unassigned"]
        n_violations = int(df["violation"].sum())
        reliability = 1.0 - n_violations / max(n_tasks, 1)
        reliability_penalty = float(n_violations)  # hard penalty

        # advance/delay profile, measured in DAY-UNITS of shift from the
        # optimal (natural) time t_n. A task left at t_n (shift 0) incurs no
        # penalty; moving it earlier (advance) or later (delay) costs.
        df["shift"] = df["scheduled_day"] - df["t_n"]
        # day-units of advance (scheduled before t_n) and delay (after t_n)
        advance_days = float((-df.loc[df["shift"] < 0, "shift"]).sum())
        delay_days = float(df.loc[df["shift"] > 0, "shift"].sum())
        advance = float((df["shift"] < 0).sum())   # #tasks advanced
        delay = float((df["shift"] > 0).sum())      # #tasks delayed

        # fitness (lower is better).
        # fitness = cost + lambda * (C_a * advance_days + C_d * delay_days)
        #   - cost          : deployment (transport + personnel) ~ cost_per_service * #clusters
        #   - shift penalty : asymmetric; delaying (C_d) costs more than advancing (C_a)
        #                     because a unit pushed past its nominal service time
        #                     has higher failure risk before the repair.
        # Reliability is a HARD safety requirement: a schedule with ANY violation
        # is disqualified (fitness -> infinity), so the optimiser never accepts a
        # cheaper-but-unsafe schedule. With 0 violations the fitness has two live
        # terms (cost AND the asymmetric shift penalty), so the optimiser trades
        # off fewer-deployments against staying close to the optimal time.
        shift_penalty = self.lambda_shift * (
            self.advance_cost * advance_days + self.delay_cost * delay_days)
        if n_violations > 0:
            fitness = float("inf")
        else:
            fitness = cost + shift_penalty
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
            "advance_days": advance_days,
            "delay_days": delay_days,
            "shift_penalty": shift_penalty,
            "fitness": fitness,
        }