"""
Stage 1 — Maintenance plan generation (degradation simulation).

Replicates the MIMAR (Wu et al., 2025) first-stage pipeline:
  - |P| PMECs, each initialised with parameters drawn from uniform distributions:
        initial pressure P0  in (3.2, 3.5] bar
        daily degradation alpha in [0.01, 0.05] bar/day
        k = number of GR interventions allowed before a RE is required, in {3,4,5}
  - Pressure decreases linearly by `alpha` per day under normal operation.
  - When pressure <= 3.2 bar  -> a Gas Refilling (GR) task is triggered.
  - GR restores pressure linearly back to 3.5 bar within one day.
  - The planning horizon T (days) is divided into sliding windows of length L.

The output is a maintenance schedule (data frame) that is the INPUT to Stage 2
(grouping ILP).

This module deliberately mirrors the paper's parameter table (Table 2) so that
Stage 2 can be checked against the paper's reported numbers
(22 tasks -> 7 clusters, 220 -> 70 cost units, 68.2% reduction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- Paper parameter table (Table 2) --------------------------------------
P_MAX = 3.5          # bar, pressure after a GR / after RE
P_GR_TRIGGER = 3.2   # bar, pressure that triggers a Gas Refilling
P_CRITICAL = 3.0     # bar, low-pressure alarm threshold (safety floor)
P_MAX_GR = 5         # tasks per cluster (C^max_GR)
T = 30              # planning horizon (days), sampling = 1 day
L = 10             # sliding window length (days)
A = 3             # max days a task may be advanced
SAFETY_MARGIN = 2  # conservative safety days before reaching P_CRITICAL
N_PMEC = 10       # |P| number of units
COST_PER_GR = 10  # cost units per deployment (transport + personnel)


def simulate_unit(unit_id: int,
                  p0: float,
                  alpha: float,
                  k: int,
                  horizon: int = T) -> list[dict]:
    """Simulate one PMEC's GR schedule over the planning horizon.

    Returns a list of GR-task records: {unit, task_id, t_n (day), alpha, k_used}
    """
    tasks = []
    pressure = p0
    day = 0
    gr_count = 0  # number of GRs executed since the last (RE) reset
    task_id = 0
    while day < horizon:
        # linear degradation over the day
        pressure -= alpha
        if pressure <= P_GR_TRIGGER:
            # GR triggered -> record a task at this day
            task_id += 1
            tasks.append({
                "unit": unit_id,
                "task_id": task_id,
                "t_n": day,
                "alpha": alpha,
                "k_used": gr_count + 1,
            })
            gr_count += 1
            # after enough GRs, a RE would be required; for the GR-grouping
            # study we keep simulating but reset the GR counter at RE points.
            if gr_count >= k:
                gr_count = 0  # RE restores to near-new; continue as GR cycle
            # GR restores pressure back to P_MAX within one day
            pressure = P_MAX
        day += 1
    return tasks


def generate_maintenance_plan(seed: int = 0) -> pd.DataFrame:
    """Generate the full maintenance plan across |P| PMECs (Stage 1 output)."""
    rng = np.random.default_rng(seed)
    all_tasks = []
    for unit in range(1, N_PMEC + 1):
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
    print(f"Generated {len(df)} GR tasks across {N_PMEC} PMECs over {T} days")  # noqa
    print(df.head(20).to_string(index=False))
    print(f"\nTasks per unit:\n{df.groupby('unit').size().to_string()}")