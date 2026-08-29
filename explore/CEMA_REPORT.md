# Constraint-Evolving Multi-Agent (CEMA) — An Exploration

*A new exploration, separate from the existing strategy-evolution system.
All code lives in `explore/`; the existing `agents/`, `reproduce/`,
`validate/`, `evaluate/` code is **untouched**.*

## 1. Motivation

The existing multi-agent system evolves a **strategy** (parameters) against
a **fixed constraint set**. This exploration does the complementary
thing: it keeps the **objective fixed** and lets the multi-agent system
**find and evolve the CONSTRAINTS** that achieve it. The hypothesis is
that, left to explore, the system will discover interesting constraints —
some that help, some that hurt, some that are infeasible (a finding in
itself).

## 2. The CEMA architecture

The evolution object is a **set of constraints** (not a parameter bundle,
not a solution). A goal is fixed; the system proposes, tests, keeps /
discards / combines constraints to achieve it.

| Agent | Role |
|---|---|
| **Goal agent** (fixed) | holds the goal: minimise **total cost** with 0 reliability violations |
| **Constraint proposer** | proposes candidate constraints (heuristic templates + an LLM) |
| **Constraint evaluator** | adds a candidate constraint to the problem, solves it, measures feasibility / cost / interestingness |
| **Constraint evolution master** | evolves the constraint *set* (add / mutate / delete / cross-over), keeps the best, records a reflection |

## 3. The constraint family (exploration space)

Seven heuristic templates, each an addable, evolvable constraint:

| Constraint | Meaning |
|---|---|
| `group_affinity` | tasks that share a group must be on the same day |
| `group_capacity` | at most k same-group tasks per day |
| `advance_cap` | no task advanced more than A days early |
| `delay_cap` | no task delayed more than D days past its original time |
| `cohesion` | tasks whose group shares an area prefix must be co-scheduled |
| `spread` | at most k tasks per day (forces spreading) |
| `early_first` | tasks ordered by original time (first-come-first-served) |

The LLM proposer (`explore/llm_constraint_proposer.py`) can propose
additional constraints beyond these templates.

## 4. The goal (fixed)

**Total cost** = deployment cost + fluid-leakage cost +
reliability penalty:
$$\text{total} = c_{\mathrm{dep}}\cdot N_{\mathrm{cl}} + N_{\mathrm{GR}}
   + \beta_{\mathrm{leak}}\sum_n \max(x_n - t_n,0)
   + \beta_{\mathrm{rel}}\,V,$$
where $x_n$ is the scheduled day of task $n$, $t_n$ its original time,
$V$ the number of reliability violations. The base optimisation
minimises clusters with a small advance penalty so the optimum is
**unique and reproducible**.

## 5. Results (real data, 48 service tasks, $H=30$)

### 5.1 Baseline
10 clusters, deploy cost 148, total cost 23 658 (with the advance
penalty), 0 violations.

### 5.2 Single constraints vs baseline

| Constraint | Feasible | Total cost | Δ vs baseline | Classification |
|---|---|---|---|---|
| `group_affinity` | **No** | infeasible | — | infeasible |
| `group_capacity` | Yes | 27 658 | +4 000 | hurts |
| `advance_cap` | Yes | 23 658 | 0 | neutral |
| `delay_cap` | Yes | 17 658 | **−6 000** | **helps** |
| `cohesion` | **No** | infeasible | — | infeasible |
| `spread` | Yes | 23 658 | 0 | neutral |
| `early_first` | Yes | 24 158 | +500 | hurts |

### 5.3 What the evolution found
The CEMA genetic algorithm converged on **`delay_cap`** as the best
constraint set: it improves the total cost by **−6 000** vs the
baseline (by limiting how far tasks may be delayed, which keeps the
schedule tight and avoids the leakage / delay penalty), with **0
reliability violations**.

### 5.4 Interesting findings
1. **`delay_cap` helps** — a *restrictive* constraint (limiting delay)
   actually lowers the total cost on this instance. A non-obvious
   result: "be conservative about delaying" is a good constraint here.
2. **`group_affinity` and `cohesion` are infeasible** — you cannot
   co-schedule all same-group tasks within the time windows / capacity.
   Discovering an infeasible constraint is itself a finding (it tells
   the planner these "nice-to-have" groupings are impossible).
3. **`group_capacity` and `early_first` hurt** — forcing group limits /
   order worsens the cost.
4. **`advance_cap` and `spread` are neutral** — they don't change the
   outcome on this instance.

## 6. Comparison with the strategy-evolution system

| | Strategy-evolution (existing) | Constraint-evolution (CEMA, this) |
|---|---|---|
| Evolution object | strategy parameters | **constraint set** |
| What is fixed | constraints | **objective** |
| Key result | 6 clusters / 87.5% / 0 viol | `delay_cap` −6 000 total cost / 0 viol |
| Finds | better schedules | better / worse / infeasible **constraints** |

## 7. Where CEMA is good and where it is not (research view)

**Good.** CEMA is good at *discovering which constraints help / hurt /
are infeasible* for a fixed goal — a job a parameter-evolution cannot
do. It found a genuinely useful constraint (`delay_cap`) and two
impossible ones, neither of which a human would have guessed a priori.

**Bad.** CEMA is not good at *guaranteeing optimality* of the final
schedule (it is a heuristic search over constraint sets); and on this
small instance the gain is modest. It is most valuable as an *exploratory*
tool to surface interesting constraints, not as a production scheduler.

## 8. Reproducibility

All CEMA results are deterministic (single-threaded CP-SAT with a
unique-objective formulation). See `explore/analysis.py`
→ `explore/cea_analysis.json`.