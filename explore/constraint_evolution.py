"""
ConstraintEvolutionMaster -- the CEMA engine.

The ONLY agent that changes the constraint set. The evolution object here
is a SET OF CONSTRAINTS (not a parameter bundle, not a solution). A goal is
FIXED ("minimise total cost, 0 reliability violations"); the master runs a
genetic algorithm over constraint sets, evaluating each with the
ConstraintEvaluator, and keeps/breeds the best.

This is a NEW exploration module; it does not import or modify the
existing agents/ strategy-evolution system.
"""
from __future__ import annotations
import copy
import json
import time
import numpy as np
import pandas as pd
from explore.constraint_evaluator import evaluate_constraint, interestingness
from explore.constraint_families import all_template_constraints


class ConstraintSet:
    """A set of constraints -- the evolvable genome in CEMA."""
    def __init__(self, constraints: list):
        self.constraints = list(constraints)
        self.metrics: dict = {}
        self.fitness: float = 0.0

    def active(self) -> list:
        return [c for c in self.constraints if getattr(c, "active", True)]

    def describe(self) -> str:
        return ",".join(c.name for c in self.constraints if getattr(c, "active", True)) or "none"

    def mutated(self, rng) -> "ConstraintSet":
        c = ConstraintSet([copy.deepcopy(x) for x in self.constraints])
        # with some probability, add a new random template constraint
        if rng.random() < 0.3:
            pool = all_template_constraints()
            c.constraints.append(pool[rng.integers(0, len(pool))])
        # mutate / toggle existing
        for x in c.constraints:
            x = x.mutated(rng)
        # occasionally drop one
        if len(c.constraints) > 1 and rng.random() < 0.2:
            c.constraints.pop(rng.integers(0, len(c.constraints)))
        return c

    def crossover(self, other: "ConstraintSet", rng) -> "ConstraintSet":
        a = [copy.deepcopy(x) for x in self.constraints]
        b = [copy.deepcopy(x) for x in other.constraints]
        out = ConstraintSet([])
        for x in a + b:
            if rng.random() < 0.5:
                out.constraints.append(x)
        # never produce an empty set: fall back to one constraint
        if not out.constraints and (a or b):
            src = a or b
            out.constraints = [copy.deepcopy(src[0])]
        return out


class ConstraintEvolutionMaster:
    def __init__(self, tasks: pd.DataFrame, H: int, pop_size: int = 6,
                 generations: int = 12, seed: int = 0, mode: str = "ga",
                 llm_proposer=None, time_limit: float = 6.0):
        self.tasks = tasks
        self.H = H
        self.pop_size = pop_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.llm_proposer = llm_proposer
        self.time_limit = time_limit
        self.history = []
        self.best = None
        self.memory = []
        self.llm_calls = 0

    def _fitness(self, cs: ConstraintSet) -> float:
        """fitness = cost, with infeasible sets heavily penalised."""
        m = evaluate_constraint(self.tasks, cs.active(), self.H,
                                time_limit=self.time_limit)
        if not m["feasible"]:
            return 1e9
        return m["cost"]

    def _eval(self, cs: ConstraintSet) -> ConstraintSet:
        m = evaluate_constraint(self.tasks, cs.active(), self.H,
                                time_limit=self.time_limit)
        cs.metrics = m
        cs.fitness = self._fitness(cs)
        return cs

    def _initial_pop(self) -> list[ConstraintSet]:
        # seed: empty set + each single template + a couple of combos
        seeds = [ConstraintSet([])]
        for c in all_template_constraints():
            seeds.append(ConstraintSet([c]))
        # a two-constraint combo
        seeds.append(ConstraintSet([all_template_constraints()[2],
                                    all_template_constraints()[3]]))
        pop = [self._eval(s) for s in seeds]
        while len(pop) < self.pop_size:
            base = seeds[self.rng.integers(0, len(seeds))]
            pop.append(self._eval(base.mutated(self.rng)))
        pop.sort(key=lambda s: s.fitness)
        return pop

    def run(self) -> dict:
        t0 = time.time()
        pop = self._initial_pop()
        for gen in range(self.generations):
            pop.sort(key=lambda s: s.fitness)
            best = pop[0]
            self.history.append({"gen": gen, "best_fitness": best.fitness,
                                 "best_metrics": best.metrics,
                                 "best_set": best.describe()})
            # record reflection: what changed vs previous best
            prev = self.best
            self.best = best
            if prev is not None and best.fitness < prev.fitness:
                added = [c.name for c in best.active()
                         if c.name not in [x.name for x in prev.active()]]
                self.memory.append({"gen": gen,
                    "improvement": round(prev.fitness - best.fitness, 1),
                    "set": best.describe(), "metrics": best.metrics,
                    "added": added})
            elif prev is None:
                self.memory.append({"gen": gen, "set": best.describe(),
                                    "metrics": best.metrics})
            # --- next generation ---
            next_pop = [copy.deepcopy(best)]
            if self.mode == "llm" and self.llm_proposer is not None:
                report = {"constraint_set": best.describe(),
                           "metrics": best.metrics}
                proposals = self.llm_proposer.propose(gen, report, self.history)
                self.llm_calls += 1
                # proposals may be dicts (CEMA LLM proposer) or Strategy
                # objects (strategy-evolution LLM proposer); handle both.
                for p in proposals:
                    if hasattr(p, "genome"):
                        # a Strategy: map its genome to a template constraint
                        g = p.genome()
                        kind = "advance_cap" if g.get("advance_limit", 3) < 3 else \
                               "delay_cap"
                        match = [c for c in all_template_constraints()
                                 if c.kind == kind or kind in c.name]
                    else:
                        kind = p.get("kind", "") if isinstance(p, dict) else ""
                        match = [c for c in all_template_constraints()
                                 if kind == c.kind or kind in c.name]
                    if match:
                        next_pop.append(self._eval(ConstraintSet(match[:1])))
            while len(next_pop) < self.pop_size:
                p1 = pop[self.rng.integers(0, len(pop))]
                p2 = pop[self.rng.integers(0, len(pop))]
                if self.rng.random() < 0.4:
                    child = p1.crossover(p2, self.rng)
                else:
                    child = p1.mutated(self.rng)
                next_pop.append(self._eval(child))
            next_pop.sort(key=lambda s: s.fitness)
            pop = next_pop[:self.pop_size]
        return {"best": self.best, "history": self.history,
                "memory": self.memory, "elapsed": time.time() - t0,
                "llm_calls": self.llm_calls}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    tasks = pd.read_csv("results/real_tasks.csv")
    H = int(tasks["t_n"].max()) + 3
    em = ConstraintEvolutionMaster(tasks=tasks, H=H, pop_size=6,
                                   generations=10, seed=0, mode="ga")
    res = em.run()
    b = res["best"]
    print(f"\nBEST constraint set: {b.describe()}")
    print(f"  metrics: {b.metrics}")
    for h in res["history"]:
        print(f"  gen {h['gen']}: fitness={h['best_fitness']:.0f} set={h['best_set']}")
    # save for the analysis step
    json.dump({"best_set": b.describe(), "best_metrics": b.metrics,
              "history": res["history"], "memory": res["memory"],
              "elapsed": res["elapsed"]},
             open("explore/cea_result.json", "w"), indent=2)
    print("\nsaved explore/cea_result.json")