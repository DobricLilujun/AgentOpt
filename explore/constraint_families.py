"""
Constraint families for the Constraint-Evolving Multi-Agent (CEMA) exploration.

This is a NEW exploration, separate from the existing strategy-evolution
system. Here the EVOLUTION OBJECT is a SET OF CONSTRAINTS, not a parameter
bundle. A goal (e.g. "minimise total cost with 0 reliability violations")
is FIXED; the system proposes, tests, keeps/discards/combines CONSTRAINTS
to achieve it.

Each constraint is an object that knows how to:
  * describe itself,
  * be added to a CP-SAT model (given the task set + assignment variables),
  * be mutated / crossed over (so it can evolve).

The constraints below are the heuristic "template" family. The LLM proposer
(explore/llm_constraint_proposer.py) can propose additional ones.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Constraint:
    """A single, addable constraint with an evolving genome (params)."""
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    # whether the constraint is currently "on"
    active: bool = True

    # ---- interface the evolution engine uses ----
    def describe(self) -> str:
        return f"{self.name}({self.kind}): {self.params}"

    def is_feasible(self, model, tasks, x, H) -> bool:
        """Return True if this constraint can be added without making the
        problem infeasible (the evaluator probes this via solving)."""
        return True  # default; the evaluator determines feasibility by solving

    # the actual CP-SAT application is done in the evaluator, via
    # apply(); each subclass implements apply().
    def apply(self, model, tasks, x, H) -> None:
        raise NotImplementedError

    # ---- evolution operators (the constraint genome = params + active) ----
    def mutated(self, rng) -> "Constraint":
        c = copy.deepcopy(self)
        # toggle on/off
        if rng.random() < 0.25:
            c.active = not c.active
        # perturb a numeric param
        for k, v in list(c.params.items()):
            if rng.random() < 0.3:
                if isinstance(v, int):
                    c.params[k] = int(np.clip(v + rng.choice([-1, 1]) *
                        rng.integers(1, 3), 1, 8))
                elif isinstance(v, float):
                    c.params[k] = float(np.clip(v + rng.normal(0, 0.3), -2.0, 2.0))
        return c

    @classmethod
    def crossover(cls, a: "Constraint", b: "Constraint") -> "Constraint":
        # inherit the kind from a, blend params from a and b
        out = copy.deepcopy(a)
        for k, v in b.params.items():
            if np.random.rand() < 0.5:
                out.params[k] = v
        return out


# ---------------------------------------------------------------------------
# Concrete constraint templates
# ---------------------------------------------------------------------------

@dataclass
class GroupAffinityConstraint(Constraint):
    """Tasks that share a group MUST be on the same day (cluster).
    'a hard grouping' constraint."""
    def apply(self, model, tasks, x, H):
        by_group = {}
        for i in range(len(tasks)):
            by_group.setdefault(tasks["group"].iloc[i], []).append(i)
        for g, idxs in by_group.items():
            if len(idxs) > 1 and self.active:
                ref = idxs[0]
                for j in idxs[1:]:
                    model.Add(x[j] == x[ref])


@dataclass
class GroupCapacityConstraint(Constraint):
    """At most `k` tasks from the same group may be on any single day."""
    k: int = 2
    def apply(self, model, tasks, x, H):
        by_group = {}
        for i in range(len(tasks)):
            by_group.setdefault(tasks["group"].iloc[i], []).append(i)
        for g, idxs in by_group.items():
            for d in range(H + 1):
                on = [model.NewBoolVar(f"gc_{g}_{d}_{i}") for i in idxs]
                for i, z in zip(idxs, on):
                    model.Add(x[i] == d).OnlyEnforceIf(z)
                    model.Add(x[i] != d).OnlyEnforceIf(z.Not())
                model.Add(sum(on) <= self.k)


@dataclass
class AdvanceCapConstraint(Constraint):
    """No task may be advanced by more than `A` days before its original time
    t_n (x_i - t_n <= A)."""
    A: int = 3
    def apply(self, model, tasks, x, H):
        for i in range(len(tasks)):
            if self.active:
                model.Add(x[i] >= tasks["t_n"].iloc[i] - self.A)


@dataclass
class DelayCapConstraint(Constraint):
    """No task may be delayed by more than `D` days past t_n
    (x_i - t_n <= D)."""
    D: int = 10
    def apply(self, model, tasks, x, H):
        for i in range(len(tasks)):
            if self.active:
                model.Add(x[i] <= tasks["t_n"].iloc[i] + self.D)


@dataclass
class CohesionConstraint(Constraint):
    """Tasks whose group shares a common area prefix (e.g. 'AVOI5') should be
    grouped together: they must be on the same day. A 'neighbouring'
    constraint that can be *helpful* if the cost rewards cohesion."""
    prefix_len: int = 5
    def apply(self, model, tasks, x, H):
        by_pref = {}
        for i in range(len(tasks)):
            pref = str(tasks["group"].iloc[i])[:self.prefix_len]
            by_pref.setdefault(pref, []).append(i)
        for p, idxs in by_pref.items():
            if len(idxs) > 1 and self.active:
                ref = idxs[0]
                for j in idxs[1:]:
                    model.Add(x[j] == x[ref])


@dataclass
class SpreadConstraint(Constraint):
    """No more than `k` tasks may be scheduled on any single day (an
    upper bound distinct from the capacity; here it forces spreading)."""
    k: int = 6
    def apply(self, model, tasks, x, H):
        for d in range(H + 1):
            if self.active:
                on = [model.NewBoolVar(f"sp_{d}_{i}") for i in range(len(tasks))]
                for i, z in enumerate(on):
                    model.Add(x[i] == d).OnlyEnforceIf(z)
                    model.Add(x[i] != d).OnlyEnforceIf(z.Not())
                model.Add(sum(on) <= self.k)


@dataclass
class EarlyFirstConstraint(Constraint):
    """Tasks with an earlier original time t_n must be scheduled no later
    than tasks with a later t_n (preserve the 'first come, first served'
    order by original time)."""
    def apply(self, model, tasks, x, H):
        order = sorted(range(len(tasks)), key=lambda i: tasks["t_n"].iloc[i])
        for a in range(len(order) - 1):
            i, j = order[a], order[a + 1]
            if self.active:
                model.Add(x[i] <= x[j])


def all_template_constraints() -> list[Constraint]:
    """The full heuristic constraint family (the exploration space)."""
    return [
        GroupAffinityConstraint(name="group_affinity", kind="group_affinity"),
        GroupCapacityConstraint(name="group_capacity", kind="group_capacity", params={"k": 2}),
        AdvanceCapConstraint(name="advance_cap", kind="advance_cap", params={"A": 3}),
        DelayCapConstraint(name="delay_cap", kind="delay_cap", params={"D": 10}),
        CohesionConstraint(name="cohesion", kind="cohesion", params={"prefix_len": 5}),
        SpreadConstraint(name="spread", kind="spread", params={"k": 6}),
        EarlyFirstConstraint(name="early_first", kind="order"),
    ]


if __name__ == "__main__":
    for c in all_template_constraints():
        print(f"  - {c.describe()}")