"""
Ablation studies for the self-evolving multi-agent system.

Each ablation changes/disables one component and measures the effect on the
final outcome. All are run on the REAL data (48 service tasks, H=30).

  A1. Evolution ON vs OFF (evolve vs fixed conventional strategy)
  A2. Prediction source: real-data fitted rates vs simulated
  A3. Evaluation objective: multi-objective vs single-objective (safety only)
  A4. Grouping solver: ILP vs greedy heuristic
  A5. Evolution engine: GA vs LLM-driven
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from agents import (GroupingAgent, EvaluationAgent, EvolutionMaster)

REAL = "results/real_tasks.csv"


def _H() -> int:
    return int(pd.read_csv(REAL)["t_n"].max()) + 3


def _pop_with_weights(em, w_reliability: float) -> list:
    """Build an initial population for em, but force the reliability weight
    (and freeze it during evolution so the objective is fixed)."""
    from agents.evolution import Strategy
    seeds = [
        Strategy(method="ilp", C_max=5, advance_limit=3, safety_margin=2,
                 advance_prefer=0.0, w_reliability=w_reliability),
        Strategy(method="greedy", C_max=5, advance_limit=3, safety_margin=2,
                 advance_prefer=0.0, w_reliability=w_reliability),
        Strategy(method="ilp", C_max=6, advance_limit=4, safety_margin=1,
                 advance_prefer=0.5, w_reliability=w_reliability),
        Strategy(method="ilp", C_max=5, advance_limit=3, safety_margin=3,
                 advance_prefer=-0.5, w_reliability=w_reliability),
        Strategy(method="ilp", C_max=8, advance_limit=4, safety_margin=2,
                 advance_prefer=0.0, w_reliability=w_reliability),
    ]
    pop = [em._eval(s) for s in seeds]
    while len(pop) < em.pop_size:
        base = seeds[em.rng.integers(0, len(seeds))]
        child = base.mutated(em.rng)
        # freeze the objective weight so this run stays single/multi-objective
        child.w_reliability = w_reliability
        pop.append(em._eval(child))
    pop.sort(key=lambda s: s.fitness)
    return pop


def _evolve(pop_size: int = 6, generations: int = 12) -> dict:
    tasks = pd.read_csv(REAL)
    em = EvolutionMaster(tasks=tasks, pop_size=pop_size, generations=generations,
                         seed=0, mode="ga", H=_H())
    return em.run()["best_strategy"].metrics


def ablation_A_evolution() -> dict:
    """A1: evolution ON (full GA) vs OFF (fixed conventional ILP strategy)."""
    tasks = pd.read_csv(REAL)
    H = _H()
    # OFF = the conventional fixed single-objective ILP strategy
    g = GroupingAgent(method="ilp", advance_prefer=0.0, cost_per_service=10, time_limit=15)
    r = g.group(tasks, C_max=5, H=H)
    m_off = EvaluationAgent().evaluate(tasks, r, H=H)
    m_on = _evolve()
    return {"ON (self-evolution)": m_on, "OFF (fixed conventional ILP)": m_off}


def ablation_B_prediction() -> dict:
    """A2: prediction source real-data fitted vs simulated."""
    tasks = pd.read_csv(REAL)
    H = _H()
    g = GroupingAgent(method="ilp", advance_prefer=0.0, cost_per_service=10, time_limit=15)
    r_real = g.group(tasks, C_max=5, H=H)
    m_real = EvaluationAgent().evaluate(tasks, r_real, H=H)
    rng = np.random.default_rng(0)
    sim = tasks.copy()
    sim["t_n"] = rng.normal(loc=15, scale=8, size=len(sim)).clip(0, H - 1)
    r_sim = g.group(sim, C_max=5, H=H)
    m_sim = EvaluationAgent().evaluate(sim, r_sim, H=H)
    return {"real-data fitted": m_real, "simulated": m_sim}


def ablation_C_evaluation() -> dict:
    """A3: relaxed vs strict safety (evolve to a safety policy)."""
    tasks = pd.read_csv(REAL)
    H = _H()
    from agents.evolution import Strategy
    # relaxed: evolve with a low reliability (safety) penalty weight
    em1 = EvolutionMaster(tasks=tasks, pop_size=6, generations=10, seed=0, mode="ga", H=H)
    # force the initial population by rewriting seeds' weight
    def _force_relaxed(em):
        em._initial_pop = lambda: _pop_with_weights(em, w_reliability=1.0)
    def _force_strict(em):
        em._initial_pop = lambda: _pop_with_weights(em, w_reliability=5.0)
    _force_relaxed(em1)
    m1 = em1.run()["best_strategy"].metrics
    # strict: default (strong reliability penalty)
    em2 = EvolutionMaster(tasks=tasks, pop_size=6, generations=10, seed=0, mode="ga", H=H)
    _force_strict(em2)
    m2 = em2.run()["best_strategy"].metrics
    return {"relaxed safety": m1,
            "strict safety (default)": m2}


def ablation_D_grouping() -> dict:
    """A4: grouping solver ILP vs greedy."""
    tasks = pd.read_csv(REAL)
    H = _H()
    g_ilp = GroupingAgent(method="ilp", advance_prefer=0.0, cost_per_service=10, time_limit=15)
    g_greedy = GroupingAgent(method="greedy", advance_prefer=0.0, cost_per_service=10)
    r_ilp = g_ilp.group(tasks, C_max=5, H=H)
    r_greedy = g_greedy.group(tasks, C_max=5, H=H)
    m_ilp = EvaluationAgent().evaluate(tasks, r_ilp, H=H)
    m_greedy = EvaluationAgent().evaluate(tasks, r_greedy, H=H)
    return {"ILP (exact)": m_ilp, "greedy (heuristic)": m_greedy}


def ablation_E_engine() -> dict:
    """A5: GA vs LLM-driven evolution (does the LLM mode match the GA?)."""
    tasks = pd.read_csv(REAL)
    H = _H()
    em_ga = EvolutionMaster(tasks=tasks, pop_size=6, generations=10, seed=0,
                            mode="ga", H=H)
    m_ga = em_ga.run()["best_strategy"].metrics
    import agents.llm_proposer as llm_proposer
    em_llm = EvolutionMaster(tasks=tasks, pop_size=6, generations=10, seed=0,
                             mode="llm", llm_proposer=llm_proposer, H=H)
    m_llm = em_llm.run()["best_strategy"].metrics
    return {"GA": m_ga, "LLM-driven": m_llm}


def _summary(label: str, m: dict) -> str:
    return (f"  {label:40} clusters={m['n_clusters']:2}  "
            f"cost_red={m['cost_reduction']*100:5.1f}%  "
            f"violations={m['n_violations']}")


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for name, fn in [("A1_evolution", ablation_A_evolution),
                     ("A2_prediction", ablation_B_prediction),
                     ("A3_evaluation", ablation_C_evaluation),
                     ("A4_grouping", ablation_D_grouping),
                     ("A5_engine", ablation_E_engine)]:
        print(f"\n{name}")
        out[name] = fn()
        for k, v in out[name].items():
            print(_summary(k, v))
    json.dump(out, open("results/ablation.json", "w"), indent=2)
    print(f"\n[done in {time.time()-t0:.0f}s] saved results/ablation.json")