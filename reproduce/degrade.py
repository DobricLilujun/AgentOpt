"""
Stage 1 — Maintenance plan generation (degradation simulation).

Simulates a fleet of |P| hydraulic units, each initialised with parameters
drawn from uniform distributions:
        initial pressure P0  in (3.2, 3.5] bar
        daily degradation alpha in [0.01, 0.05] bar/day
        k = number of pressure-service actions allowed before an overhaul is
            required, in {3,4,5}
  - Pressure decreases linearly by `alpha` per day under normal operation.
  - When pressure <= P_SERV (3.2 bar) -> a pressure-service (repressurisation)
    task is triggered.
  - The service restores pressure linearly back to P_NOM (3.5 bar) within one
    day.
  - The planning horizon T (days) is divided into sliding windows of length L.

The output is a maintenance schedule (data frame) that is the INPUT to Stage 2
(grouping ILP).

This module mirrors the fixed model-parameter table below (the same parameters
used throughout the pipeline) so Stage 2 can be exercised end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- Model parameters ----------------------------------------------------
P_NOM = 3.5          # bar, nominal operating pressure (after a service / overhaul)
P_SERV = 3.2   # bar, pressure that triggers a pressure-service (repressurisation)
P_CRIT = 3.0     # bar, critical (safety-floor) level
C_MAX = 5         # max tasks per cluster per day (capacity)
T = 30              # planning horizon (days), sampling = 1 day
L = 10             # sliding window length (days)
A = 3             # max days a task may be advanced
SAFETY_MARGIN = 2  # conservative safety days before reaching P_CRIT
N_UNIT = 10       # |P| number of units
COST_PER_SERVICE = 10  # cost units per deployment (transport + personnel)


def simulate_unit(unit_id: int,
                  p0: float,
                  alpha: float,
                  k: int,
                  horizon: int = T) -> list[dict]:
    """Simulate one hydraulic unit's service schedule over the horizon.

    Returns a list of service-task records: {unit, task_id, t_n (day), alpha, k_used}
    """
    tasks = []
    pressure = p0
    day = 0
    service_count = 0  # number of services since the last overhaul reset
    task_id = 0
    while day < horizon:
        # linear degradation over the day
        pressure -= alpha
        if pressure <= P_SERV:
            # service triggered -> record a task at this day
            task_id += 1
            tasks.append({
                "unit": unit_id,
                "task_id": task_id,
                "t_n": day,
                "alpha": alpha,
                "k_used": service_count + 1,
            })
            service_count += 1
            # after enough services, an overhaul would be required; for the
            # grouping study we keep simulating but reset the counter at overhaul points.
            if service_count >= k:
                service_count = 0  # overhaul restores to near-new; continue as service cycle
            # the service restores pressure back to P_NOM within one day
            pressure = P_NOM
        day += 1
    return tasks


def generate_maintenance_plan(seed: int = 0) -> pd.DataFrame:
    """Generate the full maintenance plan across |P| units (Stage 1 output)."""
    rng = np.random.default_rng(seed)
    all_tasks = []
    for unit in range(1, N_UNIT + 1):
        p0 = rng.uniform(3.2, 3.5)
        alpha = rng.uniform(0.01, 0.05)
        k = int(rng.integers(3, 6))  # {3,4,5}
        all_tasks.extend(simulate_unit(unit, p0, alpha, k, horizon=T))
    df = pd.DataFrame(all_tasks)
    df = df.rename(columns={"task_id": "task_global"})
    df = df.reset_index(drop=True)
    # global unique task id used as the assignment key in Stage 2
    df = df.rename(columns={"index": "tid"}).drop(columns="tid", errors="ignore")
    df["tid"] = np.arange(len(df))
    return df


if __name__ == "__main__":
    df = generate_maintenance_plan(seed=0)
    print(f"Generated {len(df)} service tasks across {N_UNIT} units over {T} days")  # noqa
    print(df.head(20).to_string(index=False))
    print(f"\nTasks per unit:\n{df.groupby('unit').size().to_string()}")