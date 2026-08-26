"""
Run the LLM-driven self-evolution mode and compare it to the GA baseline.

The EvolutionMaster in mode="llm" asks an LLM (via `hermes -z`) to PROPOSE
candidate strategies each generation; the EvaluationAgent scores them. This is
a genuine LLM-guided evolution. We run it on the real data and report the
convergence + best strategy, then compare to the GA result.
"""
from __future__ import annotations

import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubadmin/projects/AgentOpt")
from agents import EvolutionMaster
from agents import llm_proposer

OUT = "/home/ubadmin/projects/AgentOpt/results"


def run_llm(tasks: pd.DataFrame, generations: int = 10) -> dict:
    print(f"Running LLM-driven evolution: {generations} generations, "
          f"pop_size=6 (real tasks={len(tasks)})...")
    em = EvolutionMaster(tasks=tasks, pop_size=6, generations=generations,
                         seed=0, mode="llm", llm_proposer=llm_proposer)
    t0 = time.time()
    result = em.run()
    print(f"  done in {time.time()-t0:.0f}s; LLM calls made = {result['llm_calls']}")
    return result


def main():
    tasks = pd.read_csv(f"{OUT}/real_tasks.csv")

    llm = run_llm(tasks, generations=10)
    b = llm["best_strategy"]
    print("\n=== LLM-DRIVEN EVOLUTION RESULT ===")
    print("best genome:", b.genome())
    print("best metrics:", b.metrics)
    print("\nGeneration history (best clusters / cost-reduction / viol):")
    for h in llm["history"]:
        m = h["best_metrics"]
        print(f"  gen {h['gen']:2d}: clusters={m.get('n_clusters')} "
              f"cost_red={m.get('cost_reduction',0):.1%} viol={m.get('n_violations')}")

    out = {
        "mode": "llm",
        "llm_calls": llm["llm_calls"],
        "best_strategy": b.genome(),
        "best_metrics": b.metrics,
        "history": llm["history"],
        "memory": llm["memory"],
    }
    with open(f"{OUT}/llm_evolution.json", "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"\nsaved {OUT}/llm_evolution.json")


if __name__ == "__main__":
    main()