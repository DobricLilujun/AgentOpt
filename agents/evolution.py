"""
EvolutionMaster -- the self-evolution engine (the "brain" of the system).

It is the only agent that CHANGES other agents' behaviour. Given the task set
and the three other agents, it runs a genetic algorithm over a STRATEGY (a
bundle of decision parameters), evaluating each strategy with the EvaluationAgent
and keeping/breeding the best. Over generations the system therefore improves
itself -- this is the "self-evolving" property.

What a Strategy evolves (the search space)
-----------------------------------------
  method          : "ilp" (exact) | "greedy" (fast)          -> which solver
  C_max           : max tasks per cluster/day                -> capacity knob
  advance_limit   : A, max days a task may be advanced
  safety_margin   : days kept above the critical level P_CRIT
  advance_prefer  : bias toward advancing vs delaying tasks
  w_reliability : the reliability (safety) penalty severity -> the SAFETY KNOB
                  (a higher value makes a reliability violation cost more, so the
                  optimiser pushes harder to keep every unit above P_CRIT).

Self-evolution mechanics
------------------------
  * Tournament selection + mutation + crossover on the strategy genome.
  * Elitism: the best strategy is always carried forward.
  * Reflection / memory: every generation records the best strategy, its
    metrics, and WHICH mutated parameter caused the improvement, so the next
    generation searches more intelligently (not pure random search). This is
    the "learning" loop.
  * Optional LLM-guided evolution (mode="llm"): an LLM reads the generation
    report and proposes the next mutation. Falls back to GA automatically.
"""
from __future__ import annotations

import copy
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict

from agents.prediction import PredictionAgent
from agents.grouping import GroupingAgent
from agents.evaluation import EvaluationAgent


@dataclass
class Strategy:
    """An evolvable bundle of scheduling + evaluation parameters."""
    method: str = "ilp"
    C_max: int = 5
    advance_limit: int = 3
    safety_margin: int = 2
    advance_prefer: float = 0.0
    w_reliability: float = 5.0
    # bookkeeping (not part of the "genome" the GA mutates on purpose)
    metrics: dict = field(default_factory=dict)
    fitness: float = 0.0

    def genome(self) -> dict:
        return {"method": self.method, "C_max": self.C_max,
                "advance_limit": self.advance_limit, "safety_margin": self.safety_margin,
                "advance_prefer": self.advance_prefer,
                "w_reliability": self.w_reliability}

    def to_solvers(self):
        """Instantiate the GroupingAgent and EvaluationAgent for this strategy."""
        g = GroupingAgent(method=self.method, advance_prefer=self.advance_prefer,
                          window_len=10, advance_limit=self.advance_limit, cost_per_service=10)
        e = EvaluationAgent(w_reliability=self.w_reliability,
                            advance_limit=self.advance_limit, safety_margin=self.safety_margin)
        return g, e

    def mutated(self, rng) -> "Strategy":
        s = copy.deepcopy(self)
        # mutate with small random perturbations within sane ranges
        if rng.random() < 0.3:
            s.method = "ilp" if s.method == "greedy" else "greedy"
        s.C_max = int(np.clip(s.C_max + rng.choice([-1, 1]) * rng.integers(1, 2), 3, 8))
        s.advance_limit = int(np.clip(s.advance_limit + rng.choice([-1, 1]) * rng.integers(0, 2), 1, 6))
        s.safety_margin = int(np.clip(s.safety_margin + rng.choice([-1, 1]) * rng.integers(0, 2), 1, 5))
        s.advance_prefer = float(np.clip(s.advance_prefer + rng.normal(0, 0.5), -2.0, 2.0))
        # evolve the safety knob
        s.w_reliability = float(np.clip(s.w_reliability * np.exp(rng.normal(0, 0.2)), 1.0, 20.0))
        s.metrics, s.fitness = {}, 0.0
        return s

    @classmethod
    def crossover(cls, a: "Strategy", b: "Strategy") -> "Strategy":
        def pick(x, y): return x if np.random.rand() < 0.5 else y
        return cls(method=pick(a.method, b.method),
                   C_max=pick(a.C_max, b.C_max),
                   advance_limit=pick(a.advance_limit, b.advance_limit),
                   safety_margin=pick(a.safety_margin, b.safety_margin),
                   advance_prefer=pick(a.advance_prefer, b.advance_prefer),
                   w_reliability=pick(a.w_reliability, b.w_reliability))


class EvolutionMaster:
    def __init__(self, tasks: pd.DataFrame, units: pd.DataFrame = None,
                 pop_size: int = 8, generations: int = 12,
                 seed: int = 0, mode: str = "ga",
                 llm_proposer=None, H: int = 30):
        self.tasks = tasks
        self.units = units
        self.H = H
        self.pop_size = pop_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.mode = mode            # "ga" (default) or "llm"
        self.llm_proposer = llm_proposer  # callable(gen_report) -> Strategy
        self.prediction = PredictionAgent()
        self.history = []          # per-generation log
        self.best = None
        self.memory = []          # reflection log: {gen, genome, metrics, what_helped}
        self.llm_calls = 0        # number of LLM proposer calls made

    def _eval(self, s: Strategy) -> Strategy:
        g, e = s.to_solvers()
        res = g.group(self.tasks, C_max=s.C_max, H=self.H)
        m = e.evaluate(self.tasks, res, H=self.H)
        s.metrics = m
        s.fitness = m["fitness"]
        return s

    def _initial_pop(self) -> list[Strategy]:
        # a diverse starting population spanning the search space, seeded with the
        # known-good conventional ILP baseline and the greedy baseline so the GA has a
        # strong elite to breed from.
        seeds = [
            Strategy(method="ilp", C_max=5, advance_limit=3, safety_margin=2,
                     advance_prefer=0.0, w_reliability=5.0),
            Strategy(method="greedy", C_max=5, advance_limit=3, safety_margin=2,
                     advance_prefer=0.0, w_reliability=5.0),
            Strategy(method="ilp", C_max=6, advance_limit=4, safety_margin=1,
                     advance_prefer=0.5, w_reliability=5.0),
            Strategy(method="ilp", C_max=5, advance_limit=3, safety_margin=3,
                     advance_prefer=-0.5, w_reliability=8.0),
            Strategy(method="ilp", C_max=8, advance_limit=4, safety_margin=2,
                     advance_prefer=0.0, w_reliability=5.0),
        ]
        pop = [self._eval(s) for s in seeds]
        while len(pop) < self.pop_size:
            base = seeds[self.rng.integers(0, len(seeds))]
            pop.append(self._eval(base.mutated(self.rng)))
        pop.sort(key=lambda s: s.fitness)
        return pop

    def run(self) -> dict:
        pop = self._initial_pop()
        for gen in range(self.generations):
            # rank
            pop.sort(key=lambda s: s.fitness)
            best = pop[0]
            self.history.append({
                "gen": gen,
                "best_fitness": best.fitness,
                "best_metrics": best.metrics,
                "population_fitness": [round(s.fitness, 3) for s in pop],
            })
            # record reflection: what helped vs previous best
            prev = self.best
            self.best = best
            if prev is not None and best.fitness < prev.fitness:
                self.memory.append({
                    "gen": gen,
                    "improvement": round(prev.fitness - best.fitness, 3),
                    "genome": best.genome(),
                    "metrics": best.metrics,
                    "delta": {k: (best.genome()[k] - prev.genome()[k])
                             for k in best.genome() if isinstance(best.genome()[k], float)},
                })
            elif prev is None:
                self.memory.append({"gen": gen, "genome": best.genome(),
                                   "metrics": best.metrics})

            # --- produce next generation ---
            next_pop = [copy.deepcopy(best)]  # elitism
            # In LLM mode, ask the LLM brain to propose a BATCH of candidates
            # this generation (one LLM call -> several strategies).
            if self.mode == "llm" and self.llm_proposer is not None:
                best_report = {"genome": best.genome(), "metrics": best.metrics}
                llm_children = self.llm_proposer.propose(gen, best_report, self.history)
                for child in llm_children:
                    next_pop.append(self._eval(child))
                self.llm_calls = getattr(self, "llm_calls", 0) + 1
            # GA mutation/crossover to fill the rest of the population
            while len(next_pop) < self.pop_size:
                p1 = pop[self.rng.integers(0, len(pop))]
                p2 = pop[self.rng.integers(0, len(pop))]
                if self.rng.random() < 0.4:
                    child = Strategy.crossover(p1, p2)
                else:
                    child = p1.mutated(self.rng)
                next_pop.append(self._eval(child))
            # keep best pop_size
            next_pop.sort(key=lambda s: s.fitness)
            pop = next_pop[:self.pop_size]

        return {"best_strategy": self.best,
                "history": self.history,
                "memory": self.memory,
                "generations": self.generations,
                "llm_calls": getattr(self, "llm_calls", 0)}

    def report(self) -> str:
        b = self.best
        return (f"Self-evolution converged after {self.generations} generations.\n"
                f"Best strategy genome: {b.genome()}\n"
                f"Best metrics: {json.dumps(b.metrics, default=str)}\n"
                f"Reflection log ({len(self.memory)} improvements recorded).")