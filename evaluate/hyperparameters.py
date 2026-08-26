"""
Hyperparameter comparisons for the multi-agent system.

Sweeps each hyperparameter and measures the effect on the final outcome
(clusters, cost reduction, leakage, violations) on the REAL data (H=30).

  H1. population size    : 4, 6, 8, 10
  H2. generations        : 4, 8, 12, 16
  H3. C_max (capacity)   : 3, 5, 7, 8   (direct sweep of the grouping agent)
  H4. advance_limit A    : 1, 3, 6
  H5. safety_margin      : 1, 2, 4
  H6. objective weights  : cost / leakage / reliability emphasis
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, ".")
import pandas as pd
from agents import GroupingAgent, EvaluationAgent, EvolutionMaster

REAL = "results/real_tasks.csv"


def _H() -> int:
    return int(pd.read_csv(REAL)["t_n"].max()) + 3


def _evolve(pop_size: int, generations: int) -> dict:
    tasks = pd.read_csv(REAL)
    em = EvolutionMaster(tasks=tasks, pop_size=pop_size, generations=generations,
                         seed=0, mode="ga", H=_H())
    return em.run()["best_strategy"].metrics


def _single_group(C_max: int, advance_limit: int = 3, safety_margin: int = 2,
                  advance_prefer: float = 0.0) -> dict:
    """Run the ILP grouping agent once with a given capacity / advance / margin."""
    tasks = pd.read_csv(REAL)
    H = _H()
    g = GroupingAgent(method="ilp", advance_limit=advance_limit, advance_prefer=advance_prefer,
                      cost_per_gr=10, time_limit=15)
    r = g.group(tasks, C_max=C_max, H=H)
    return EvaluationAgent(advance_limit=advance_limit, safety_margin=safety_margin).evaluate(
        tasks, r, H=H)


def H1_population() -> dict:
    return {str(ps): _evolve(ps, 12) for ps in [4, 6, 8, 10]}


def H2_generations() -> dict:
    return {str(g): _evolve(6, g) for g in [4, 8, 12, 16]}


def H3_capacity() -> dict:
    return {str(c): _single_group(c) for c in [3, 5, 7, 8]}


def H4_advance_limit() -> dict:
    return {str(a): _single_group(5, advance_limit=a) for a in [1, 3, 6]}


def H5_safety_margin() -> dict:
    return {str(s): _single_group(5, safety_margin=s) for s in [1, 2, 4]}


def H6_objective_weights() -> dict:
    """Cost-only vs leakage-emphasis vs reliability-emphasis, evolved."""
    tasks = pd.read_csv(REAL)
    H = _H()
    from agents.evolution import Strategy

    def _evolve_w(w_leak: float, w_reliability: float) -> dict:
        em = EvolutionMaster(tasks=tasks, pop_size=6, generations=10, seed=0, mode="ga", H=H)
        # force the objective weights (freeze during evolution)
        seeds = [
            Strategy(method="ilp", C_max=5, advance_limit=3, safety_margin=2,
                     advance_prefer=0.0, w_cost=1.0, w_leak=w_leak, w_reliability=w_reliability),
            Strategy(method="ilp", C_max=6, advance_limit=4, safety_margin=1,
                     advance_prefer=0.5, w_cost=2.0, w_leak=w_leak, w_reliability=w_reliability),
            Strategy(method="ilp", C_max=5, advance_limit=3, safety_margin=3,
                     advance_prefer=-0.5, w_cost=0.5, w_leak=w_leak, w_reliability=w_reliability),
            Strategy(method="ilp", C_max=8, advance_limit=4, safety_margin=2,
                     advance_prefer=0.0, w_cost=3.0, w_leak=w_leak, w_reliability=w_reliability),
            Strategy(method="greedy", C_max=5, advance_limit=3, safety_margin=2,
                     advance_prefer=0.0, w_cost=1.0, w_leak=w_leak, w_reliability=w_reliability),
        ]
        orig = em._initial_pop
        def _forced():
            pop = [em._eval(s) for s in seeds]
            while len(pop) < em.pop_size:
                base = seeds[em.rng.integers(0, len(seeds))]
                c = base.mutated(em.rng)
                c.w_leak, c.w_reliability = w_leak, w_reliability
                pop.append(em._eval(c))
            pop.sort(key=lambda s: s.fitness)
            return pop
        em._initial_pop = _forced
        return em.run()["best_strategy"].metrics

    return {
        "cost only": _evolve_w(0.0, 1.0),
        "cost+leakage+reliability (default)": _evolve_w(1.0, 5.0),
        "leakage emphasis": _evolve_w(3.0, 5.0),
        "reliability emphasis": _evolve_w(1.0, 20.0),
    }


def _summary(label: str, m: dict) -> str:
    return (f"  {label:34} clusters={m['n_clusters']:2}  "
            f"cost_red={m['cost_reduction']*100:5.1f}%  "
            f"leakage={m['leakage_kg']:.5f}kg  violations={m['n_violations']}")


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for name, fn in [("H1_population", H1_population),
                     ("H2_generations", H2_generations),
                     ("H3_capacity", H3_capacity),
                     ("H4_advance_limit", H4_advance_limit),
                     ("H5_safety_margin", H5_safety_margin),
                     ("H6_objective_weights", H6_objective_weights)]:
        print(f"\n{name}")
        out[name] = fn()
        for k, v in out[name].items():
            print(_summary(k, v))
    json.dump(out, open("results/hyperparameters.json", "w"), indent=2)
    print(f"\n[done in {time.time()-t0:.0f}s] saved results/hyperparameters.json")