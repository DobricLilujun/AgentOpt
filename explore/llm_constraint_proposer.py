"""
LLMConstraintProposer -- proposes NEW candidate constraints (and parameter
changes) to an LLM, in the CEMA exploration.

The LLM reads the current best constraint set + its evaluation report and
proposes new constraints that might help the fixed goal. This is where
"interesting" (non-obvious) constraints can come from -- beyond the
heuristic template family.

Uses the `hermes -z` one-shot LLM interface (same as the strategy
evolution), so it works with the configured provider. No changes to
existing agents/.
"""
from __future__ import annotations
import re
import json
import subprocess


def _llm_call(prompt: str, timeout: int = 180) -> str:
    try:
        r = subprocess.run(["hermes", "-z", prompt], capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"ERROR: {e}"


def _parse(out: str) -> list[dict]:
    if not out or "ERROR" in out[:10]:
        return []
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [s for s in data if isinstance(s, dict)]


def _build_prompt(gen: int, best: dict, history: list) -> str:
    best_set = best.get("constraint_set", [])
    best_m = best.get("metrics", {})
    recent = [(h["gen"], round(h.get("best_fitness", 0), 2)) for h in history[-4:]]
    return (
        f"You are exploring a constraint-optimization problem for "
        f"predictive maintenance scheduling of pressure-service (repressurisation) "
        f"tasks in a hydraulic station group.\n"
        f"GOAL (fixed): minimise total deployment cost, with 0 reliability "
        f"violations (no task scheduled before it is safe).\n"
        f"The problem is a task-grouping ILP: each service task has an original "
        f"time t_n; x_i is the day it is scheduled; a cluster = a day with "
        f">=1 task; minimise number of clusters; per-day capacity = 5.\n\n"
        f"Current best constraint set: {best_set}\n"
        f"Current best metrics: {best_m}\n"
        f"Recent generation fitness (gen, fitness): {recent}\n\n"
        f"Propose up to 3 NEW constraints (or parameter changes) that might "
        f"help the goal. A constraint can be any relation among the tasks' "
        f"groups, original times, demand, or schedule days. Be creative; "
        f"interesting non-obvious constraints are welcome. "
        f"Return ONLY a JSON array of objects, each with fields:\n"
        f'  name (str), kind (str), params (dict), apply (short text).\n'
        f"Example: "
        f'[{{"name":"group_affinity","kind":"group_affinity",'
        f'"params":{{}},"apply":"same-group tasks on same day"}}]'
    )


def propose(gen: int, best: dict, history: list) -> list[dict]:
    """Ask the LLM to propose candidate constraint descriptions."""
    out = _llm_call(_build_prompt(gen, best, history))
    return _parse(out)


if __name__ == "__main__":
    # smoke test: one LLM proposal
    out = propose(1, {"constraint_set": [], "metrics": {"cost": 148}}, [])
    print("LLM proposed:", out)