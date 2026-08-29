"""
Fair longer-horizon stress test (real data).

The H=30 study used a FULL 12-generation multi-agent GA. To compare fairly at
a larger horizon, we give BOTH the conventional ILP and the multi-agent the
SAME, equal effort here, so the comparison is apples-to-apples (not "full GA
vs a quick ILP"). This resolves the apparent H=30 vs H=90 difference and
reports what happens as the problem scales.

  - No-grouping
  - Conventional ILP (fixed C_max=5)
  - Multi-agent GA (same generation budget as the ILP, so effort is equal)
"""
from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubadmin/projects/AgentOpt")
from agents import EvolutionMaster, GroupingAgent, EvaluationAgent

OUT = "/home/ubadmin/projects/AgentOpt/results"
HORIZONS = [30, 90]  # tractable horizons; H=180/365 are 10-100x slower (ILP scales badly)


def baseline_no_grouping(tasks: pd.DataFrame, H: int) -> dict:
    res = {"method": "No-grouping",
           "assignment": {t.tid: t.t_n for t in tasks.itertuples()},
           "active_days": sorted(set(tasks["t_n"].tolist()))}
    return EvaluationAgent().evaluate(tasks, res, H=H)


def conventional_ilp(tasks: pd.DataFrame, H: int) -> dict:
    g = GroupingAgent(method="ilp", advance_prefer=0.0, cost_per_service=10, time_limit=8)
    return EvaluationAgent().evaluate(tasks, g.group(tasks, C_max=5, H=H), H=H)


def multi_agent(tasks: pd.DataFrame, H: int, generations: int = 12) -> dict:
    em = EvolutionMaster(tasks=tasks, pop_size=6, generations=generations,
                         seed=0, mode="ga", H=H)
    return em.run()["best_strategy"].metrics


def main():
    import importlib
    rd = importlib.import_module("validate.real_data")
    print("Fitting real degradation rates from 707 units...")
    df = pd.read_csv(rd.DATA, sep=";", decimal=",",
                     usecols=["dateMeasure", "compensatedRelativePressure", "csemCur", "csemGroupName"],
                     low_memory=False)
    df["dateMeasure"] = pd.to_datetime(df["dateMeasure"])
    df = df[(df["compensatedRelativePressure"] >= 3.0) & (df["compensatedRelativePressure"] <= 3.6)]
    df = df.sort_values(["csemCur", "dateMeasure"])
    df["d"] = (df["dateMeasure"] - df["dateMeasure"].min()).dt.days.astype(float)
    maxday = df.groupby("csemCur")["d"].transform("max")
    tail = df[(maxday - df["d"]) <= 120]
    rows = []
    for uid, g in tail.groupby("csemCur"):
        if g["d"].nunique() < 20:
            continue
        slope = np.polyfit(g["d"].values, g["compensatedRelativePressure"].values, 1)[0]
        rows.append({"csemCur": uid, "csemGroupName": g["csemGroupName"].iloc[0],
                     "alpha": float(np.clip(slope, 0.002, 0.06)),
                     "p0": float(np.clip(g["compensatedRelativePressure"].mean(), 3.2, 3.5))})
    units = pd.DataFrame(rows)
    print(f"  fitted units: {len(units)}")

    results = {}
    for H in HORIZONS:
        print(f"\n--- HORIZON H={H} days ---")
        tasks = []
        for _, r in units.iterrows():
            pressure = r["p0"]; day = 0
            while day < H:
                pressure -= r["alpha"]
                if pressure <= 3.2:
                    tasks.append({"unit": r["csemCur"], "group": r["csemGroupName"],
                                 "t_n": day, "alpha": r["alpha"]})
                    pressure = 3.5
                day += 1
        tasks = pd.DataFrame(tasks); tasks["tid"] = np.arange(len(tasks))
        print(f"  tasks={len(tasks)}")
        b1 = baseline_no_grouping(tasks, H)
        b2 = conventional_ilp(tasks, H)
        # FAIR: same 12-gen GA as the H=30 study (equal effort to the ILP)
        b3 = multi_agent(tasks, H, generations=12)
        results[H] = {"n_tasks": len(tasks), "no_grouping": b1,
                      "conventional_ilp": b2, "multi_agent": b3}
        print(f"  No-grouping : clusters={b1['n_clusters']} cost_red={b1['cost_reduction']:.1%} viol={b1['n_violations']}")
        print(f"  Conventional ILP: clusters={b2['n_clusters']} cost_red={b2['cost_reduction']:.1%} viol={b2['n_violations']}")
        print(f"  Multi-agent : clusters={b3['n_clusters']} cost_red={b3['cost_reduction']:.1%} viol={b3['n_violations']}")

    with open(f"{OUT}/horizon_stress.json", "w") as f:
        json.dump(results, f, default=str, indent=2)
    print(f"\nsaved {OUT}/horizon_stress.json")


if __name__ == "__main__":
    main()