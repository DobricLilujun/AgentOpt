"""
Evaluation harness: compare the three approaches on the REAL data.

  1. No-grouping baseline   -- every GR task done at its original time
  2. Paper ILP             -- the MIMAR single-objective ILP (min clusters)
  3. Self-evolving agents  -- the EvolutionMaster's evolved strategy

Reported metrics: #clusters (deployments), cost & reduction, SF6 leakage,
reliability (violations), and the advance/delay profile.
"""
from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubadmin/projects/AgentOpt")
from agents import EvolutionMaster, GroupingAgent, EvaluationAgent


def baseline_no_grouping(tasks: pd.DataFrame, H: int = 30) -> dict:
    """Deploy every task at its original time (no grouping)."""
    res = {"method": "No-grouping",
           "assignment": {t.tid: t.t_n for t in tasks.itertuples()},
           "active_days": sorted(set(tasks["t_n"].tolist()))}
    e = EvaluationAgent()
    return e.evaluate(tasks, res, H=H)


def run_paper_ilp(tasks: pd.DataFrame, H: int = 30) -> dict:
    g = GroupingAgent(method="ilp", advance_prefer=0.0, cost_per_gr=10)
    e = EvaluationAgent()
    res = g.group(tasks, C_max=5, H=H)
    return e.evaluate(tasks, res, H=H)


def run_multi_agent(tasks: pd.DataFrame, H: int = 30) -> dict:
    em = EvolutionMaster(tasks=tasks, pop_size=6, generations=10, seed=0, mode="ga")
    result = em.run()
    return {"strategy": result["best_strategy"].genome(),
            "metrics": result["best_strategy"].metrics,
            "history": result["history"],
            "memory": result["memory"]}


def main():
    tasks = pd.read_csv("/home/ubadmin/projects/AgentOpt/results/real_tasks.csv")
    print(f"Real tasks: {len(tasks)}")

    b1 = baseline_no_grouping(tasks)
    b2 = run_paper_ilp(tasks)
    b3 = run_multi_agent(tasks)

    rows = {
        "No-grouping": b1,
        "Paper ILP": b2,
        "Self-evolving agents": b3["metrics"],
    }
    print("\n=== COMPARISON (real data) ===")
    header = f"{'method':<22}{'clusters':>9}{'cost':>6}{'cost_red':>10}{'leak_kg':>10}{'viol':>6}{'reliab':>8}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        print(f"{name:<22}{m['n_clusters']:>9}{m['cost']:>6}"
              f"{m['cost_reduction']:>10.1%}{m['leakage_kg']:>10.4f}"
              f"{m['n_violations']:>6}{m['reliability']:>8.2f}")

    if "strategy" in b3:
        print("\nEvolved strategy:", b3["strategy"])
        print("\nEvolution progress (cost_reduction per generation):")
        for h in b3["history"]:
            m = h["best_metrics"]
            print(f"  gen {h['gen']:2d}: clusters={m.get('n_clusters'):2d} "
                  f"cost_red={m['cost_reduction']:.1%} viol={m.get('n_violations')}")

    # save
    out = {"tasks": len(tasks),
           "baselines": rows,
           "evolved_strategy": b3.get("strategy"),
           "evolution_history": b3.get("history"),
           "reflection": b3.get("memory")}
    with open("/home/ubadmin/projects/AgentOpt/results/evaluation.json", "w") as f:
        json.dump(out, f, default=str, indent=2)
    print("\nsaved results/evaluation.json")


if __name__ == "__main__":
    main()