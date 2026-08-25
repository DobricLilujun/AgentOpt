"""
Real-data pipeline: derive a grounded maintenance schedule from the REAL
monitoring data (monitoring_PSEM_brute.csv, 707 units, 2021-05 -> 2023-05).

The real data is a HEALTHY fleet: units sit near 3.4 bar for the whole window
with almost no refills (no sawtooth), and occasional -1.01 values are sensor
artifacts, not real low pressure. So we do NOT read literal GR events; instead
we fit a REAL degradation rate alpha_i from each unit's pressure trend and use
it as the grounded input to the grouping model.

Steps
-----
1. Read the CSV (fast chunked read; European decimal + separator).
2. For each unit: keep only points with compensatedRelativePressure in [3.0, 3.6]
   (drop the -1.01 sensor artifacts). Fit a linear slope of pressure vs. days
   -> daily degradation rate alpha_i (bar/day), using the last ~120 days of
   clean data (recent degradation is what matters for the near-term horizon).
3. Build a planning horizon: for a substation (group), take its units, assume
   pressure starts at ~3.47 bar and degrades at alpha_i, and trigger a GR
   whenever pressure <= 3.2 bar, restoring to 3.5. This yields a real-grounded
   maintenance schedule to feed the grouping ILP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA = "/home/ubadmin/projects/AgentOpt/monitoring_PSEM_brute.csv"
P_MAX = 3.5
P_GR_TRIGGER = 3.2
P_VALID_LO, P_VALID_HI = 3.0, 3.6


def _to_days(dt: pd.Series) -> pd.Series:
    return (dt - dt.min()).dt.days.astype(float)


def load_degradation_rates(max_units: int = 0,
                           tail_days: int = 120,
                           horizon_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit per-unit degradation rates from the real pressure trends.

    Returns (units, tasks):
      units: per-unit degradation rate alpha and a representative start pressure
      tasks: a GR maintenance schedule built from those rates (paper horizon)
    """
    df = pd.read_csv(DATA, sep=";", decimal=",",
                     usecols=["dateMeasure", "compensatedRelativePressure", "csemCur", "csemGroupName"],
                     low_memory=False)
    df["dateMeasure"] = pd.to_datetime(df["dateMeasure"])

    # drop sensor artifacts (only keep valid pressure range)
    df = df[(df["compensatedRelativePressure"] >= P_VALID_LO) &
           (df["compensatedRelativePressure"] <= P_VALID_HI)]
    df = df.sort_values(["csemCur", "dateMeasure"])

    # tail window (recent degradation)
    df["d"] = _to_days(df["dateMeasure"])
    # per-unit max day
    maxday = df.groupby("csemCur")["d"].transform("max")
    tail = df[(maxday - df["d"]) <= tail_days]

    # fit slope of pressure vs day for each unit in the tail
    rows = []
    for uid, g in tail.groupby("csemCur"):
        if g["d"].nunique() < 20:
            continue
        slope = np.polyfit(g["d"].values, g["compensatedRelativePressure"].values, 1)[0]
        # clamp to a physically sane degradation band [0.002, 0.06] bar/day
        alpha = float(np.clip(slope, 0.002, 0.06))
        p_start = float(np.clip(g["compensatedRelativePressure"].mean(), 3.2, 3.5))
        group = g["csemGroupName"].iloc[0]
        rows.append({"csemCur": uid, "csemGroupName": group,
                     "alpha": alpha, "p0": p_start})
    units = pd.DataFrame(rows)
    if max_units:
        units = units.sample(max_units, random_state=0).reset_index(drop=True)

    # build a grounded GR maintenance schedule over a planning horizon
    tasks = []
    for u, r in units.iterrows():
        pressure = r["p0"]
        day = 0
        while day < horizon_days:
            pressure -= r["alpha"]
            if pressure <= P_GR_TRIGGER:
                tasks.append({"unit": r["csemCur"], "group": r["csemGroupName"],
                             "t_n": day, "alpha": r["alpha"]})
                pressure = P_MAX  # GR restores to max
            day += 1
    tasks = pd.DataFrame(tasks)
    tasks["tid"] = np.arange(len(tasks))
    return units, tasks


if __name__ == "__main__":
    units, tasks = load_degradation_rates(max_units=0, horizon_days=30)
    print(f"units with fitted degradation: {len(units)}")
    print(f"units per group: {units.groupby('csemGroupName').size().describe()}")
    print(f"\nfit degradation rate alpha (bar/day):")
    print(units["alpha"].describe().round(4))
    print(f"\nGR tasks in horizon: {len(tasks)}")
    print("first 10 tasks:")
    print(tasks.head(10).to_string(index=False))
    units.to_csv("/home/ubadmin/projects/AgentOpt/results/real_units.csv", index=False)
    tasks.to_csv("/home/ubadmin/projects/AgentOpt/results/real_tasks.csv", index=False)
    print("\nsaved results/real_units.csv and results/real_tasks.csv")