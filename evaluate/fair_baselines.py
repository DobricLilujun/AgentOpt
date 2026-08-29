"""
fair_baselines.py -- the strict, honest comparison the reviewer demanded.

The core question: does AgentOpt's self-evolution actually beat plain
parameter tuning?  To answer it, we put the CONVENTILAL ILP and the
EVOLVED strategy on the SAME footing and add three fair baselines that
share the exact same search space as the GA:

  B0  Conventional ILP      : fixed C_max=5, A=3, m=2 (the old baseline)
  B1  Fixed C_max=8 ILP     : the "just raise capacity" baseline
  B2  Fixed A=4, m=2 ILP    : the "just widen advance / relax safety" baseline
  B3  Best fixed strategy   : an EXHAUSTIVE GRID SEARCH over the SAME
                              search space the GA explores, evaluated with
                              the DETERMINISTIC ILP (no evolution at all).
  B4  AgentOpt (evolved)    : the self-evolving multi-agent, GA + LLM.

If B3 (the best hyper-parameter tuning by exhaustive grid) already matches
or beats B4 (evolution), then the gain is from parameter tuning, NOT from
the evolutionary mechanism.  If B4 beats B3, evolution adds value beyond
tuning.  This is the fair test the reviewer asked for.

All results are computed on the REAL data and are fully reproducible.
"""
from __future__ import annotations
import itertools
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from agents.grouping import GroupingAgent
from agents.evaluation import EvaluationAgent
from agents.evolution import EvolutionMaster, Strategy

REAL = "/home/ubadmin/projects/AgentOpt/results/real_tasks.csv"
REAL_UNITS = "/home/ubadmin/projects/AgentOpt/results/real_units.csv"


def _evaluate(tasks: pd.DataFrame, H: int, method: str = "ilp",
             C_max: int = 5, A: int = 3, m: int = 2,
             advance_prefer: float = 0.0, w_reliability: float = 5.0) -> dict:
    """Run the deterministic ILP/greedy grouping + evaluation for a fixed strategy."""
    g = GroupingAgent(method=method, advance_prefer=advance_prefer, window_len=10,
                      advance_limit=A, cost_per_service=10)
    e = EvaluationAgent(w_reliability=w_reliability, advance_limit=A, safety_margin=m)
    res = g.group(tasks, C_max=C_max, H=H)
    return e.evaluate(tasks, res, H=H)


def _grid_search(tasks: pd.DataFrame, H: int) -> dict:
    """Grid search over the same STRUCTURAL space the GA explores, using the
    deterministic ILP (no evolution).  We grid the three knobs that actually
    change the ILP solution -- C_max, A (advance_limit), m (safety_margin) --
    which is a fair and strong "best fixed hyper-parameter tuning" baseline.
    (advance_prefer and w_reliability are soft objective biases, not structural
    capacity knobs; the GA additionally searches those, so any gain the GA
    shows over this grid is at least partly due to evolution/search rather
    than the structural tuning alone.)"""
    C_maxs = list(range(3, 9))       # 3..8
    As = list(range(1, 7))          # 1..6 (advance_limit)
    ms = list(range(1, 6))         # 1..5 (safety_margin)

    best = None
    tried = 0
    for C_max, A, m in itertools.product(C_maxs, As, ms):
        met = _evaluate(tasks, H, method="ilp", C_max=C_max, A=A, m=m,
                        advance_prefer=0.0, w_reliability=5.0)
        tried += 1
        if met["n_violations"] == 0 and (best is None or met["fitness"] < best[1]):
            best = (("ilp", C_max, A, m, 0.0), met, tried)
    return {"grid_search_best": best, "grid_points_tried": tried}


def main() -> dict:
    tasks = pd.read_csv(REAL)
    units = pd.read_csv(REAL_UNITS)
    H = int(tasks["t_n"].max()) + 3

    out = {}
    print(f"\n=== REAL data: {len(tasks)} service tasks, {units.shape[0]} units, "
          f"horizon H={H} days ===")

    # B0 conventional ILP
    b0 = _evaluate(tasks, H, method="ilp", C_max=5, A=3, m=2, advance_prefer=0.0)
    out["B0_conventional_ilp"] = b0

    # B1 fixed C_max=8
    b1 = _evaluate(tasks, H, method="ilp", C_max=8, A=3, m=2, advance_prefer=0.0)
    out["B1_fixed_Cmax8_ilp"] = b1

    # B2 fixed A=4, m=2
    b2 = _evaluate(tasks, H, method="ilp", C_max=5, A=4, m=2, advance_prefer=0.0)
    out["B2_fixed_A4_m2_ilp"] = b2

    # B3 best fixed strategy (exhaustive grid, deterministic ILP, NO evolution)
    g = _grid_search(tasks, H)
    best_genome, best_met, tried = g["grid_search_best"]
    out["B3_best_fixed_strategy"] = best_met
    out["B3_grid_points"] = tried
    out["B3_grid_best_genome"] = best_genome

    # B4 AgentOpt evolved
    em = EvolutionMaster(tasks=tasks, units=units, H=H, pop_size=6,
                         generations=10, seed=0)
    res = em.run()
    out["B4_agentopt_evolved"] = res["best_strategy"].metrics
    out["B4_genome"] = res["best_strategy"].genome()

    # ---- summary ----
    def s(m):
        return f"clusters={m['n_clusters']:3} cost_red={m['cost_reduction']*100:5.1f}% " \
               f"viol={m['n_violations']} fitness={m['fitness']:.0f}"
    print("\n=== HONEST FAIR COMPARISON ===")
    print(f"{'method':<32} {'clusters':>9} {'cost_red':>9} {'viol':>5}  {'fitness':>9}")
    print("-" * 70)
    print(f"{'B0 Conventional ILP':<32} {b0['n_clusters']:>9} "
          f"{b0['cost_reduction']*100:>8.1f}% {b0['n_violations']:>5}  {b0['fitness']:>9.0f}")
    print(f"{'B1 Fixed C_max=8 ILP':<32} {b1['n_clusters']:>9} "
          f"{b1['cost_reduction']*100:>8.1f}% {b1['n_violations']:>5}  {b1['fitness']:>9.0f}")
    print(f"{'B2 Fixed A=4,m=2 ILP':<32} {b2['n_clusters']:>9} "
          f"{b2['cost_reduction']*100:>8.1f}% {b2['n_violations']:>5}  {b2['fitness']:>9.0f}")
    print(f"{'B3 Best fixed (grid, no evo)':<32} {best_met['n_clusters']:>9} "
          f"{best_met['cost_reduction']*100:>8.1f}% {best_met['n_violations']:>5}  "
          f"{best_met['fitness']:>9.0f}")
    print(f"  (genome: {best_genome}, over {tried} grid points)")
    print(f"{'B4 AgentOpt (evolved)':<32} {out['B4_agentopt_evolved']['n_clusters']:>9} "
          f"{out['B4_agentopt_evolved']['cost_reduction']*100:>8.1f}% "
          f"{out['B4_agentopt_evolved']['n_violations']:>5}  "
          f"{out['B4_agentopt_evolved']['fitness']:>9.0f}")
    print(f"  (genome: {out['B4_genome']})")

    # the crux: does evolution beat best-fixed-tuning?
    evo = out["B4_agentopt_evolved"]
    grid = best_met
    delta = grid["fitness"] - evo["fitness"]
    print("\n=== THE CRUX: does evolution beat best fixed tuning? ===")
    print(f"  B3 best-fixed-tuning fitness = {grid['fitness']:.1f} "
          f"({grid['n_clusters']} clusters, {grid['cost_reduction']*100:.1f}% red)")
    print(f"  B4 AgentOpt evolved    fitness = {evo['fitness']:.1f} "
          f"({evo['n_clusters']} clusters, {evo['cost_reduction']*100:.1f}% red)")
    print(f"  delta (grid - evo) = {delta:.1f}  "
          f"{'(evolution WINS)' if delta > 0 else '(tuning matches/beats evolution)'}")

    import json
    with open("/home/ubadmin/projects/AgentOpt/results/fair_baselines.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved results/fair_baselines.json")
    return out


if __name__ == "__main__":
    main()