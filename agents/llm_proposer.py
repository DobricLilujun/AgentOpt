"""
LLMProposer -- the "brain" of the LLM-driven evolution mode.

Given a generation report (best metrics + genome so far), it asks an LLM
(through the `hermes -z` one-shot interface, which uses the same provider this
session runs on) to PROPOSE a batch of candidate strategies. Each proposed
strategy is then evaluated by the EvaluationAgent and competed against the GA
population.

This is a genuine, honest LLM-driven evolution: the LLM reads the feedback and
reasons about which parameters to change; the evaluation is still done by the
EvaluationAgent (the LLM does not get to declare its own score). It is an
extension of the paper's "behavior-specific penalties / preferences" future
work, realised with an LLM.

Efficiency: one LLM call proposes a small batch (3) of strategies per
generation, so a 10-generation run makes ~10 LLM calls -- tractable.
"""
from __future__ import annotations

import json
import subprocess
import re

from agents.evolution import Strategy

DEFAULTS = {
    "method": "ilp", "C_max": 5, "advance_limit": 3, "safety_margin": 2,
    "advance_prefer": 0.0, "w_cost": 1.0, "w_leak": 1.0, "w_reliability": 5.0,
}


def _clamp(g: dict) -> dict:
    """Clamp a proposed genome to sane ranges (the LLM may go off the rails)."""
    g = dict(g)
    g["method"] = "ilp" if g.get("method") in ("ilp", "greedy") else "ilp"
    g["C_max"] = int(max(3, min(9, int(g.get("C_max", 5)))))
    g["advance_limit"] = int(max(1, min(6, int(g.get("advance_limit", 3)))))
    g["safety_margin"] = int(max(1, min(6, int(g.get("safety_margin", 2)))))
    g["advance_prefer"] = float(max(-2.0, min(2.0, float(g.get("advance_prefer", 0.0)))))
    g["w_cost"] = float(max(0.1, min(5.0, float(g.get("w_cost", 1.0)))))
    g["w_leak"] = float(max(0.05, min(10.0, float(g.get("w_leak", 1.0)))))
    g["w_reliability"] = float(max(1.0, min(20.0, float(g.get("w_reliability", 5.0)))))
    return g


def _llm_call(prompt: str, timeout: int = 150) -> str:
    try:
        out = subprocess.run(["hermes", "-z", prompt], capture_output=True,
                             text=True, timeout=timeout)
        return out.stdout or ""
    except Exception as e:  # pragma: no cover
        return f"__ERROR__ {e}"


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
    best_g = best.get("genome", DEFAULTS)
    best_m = best.get("metrics", {})
    recent = [(h["gen"], round(h.get("best_fitness", 0), 3)) for h in history[-3:]]
    keys = ["n_clusters", "cost", "cost_reduction", "leakage_kg",
           "n_violations", "reliability"]
    metrics_json = json.dumps({k: best_m.get(k) for k in keys})
    genome_json = json.dumps(best_g)
    recent_json = json.dumps(recent)
    lines = [
        "You are the brain of a self-evolving maintenance-scheduling optimizer "
        "for gas-insulated-substation (GIS) gas-refilling (GR) tasks.",
        f"Generation {gen}. So far the best strategy (genome) and its measured metrics:",
        f"  genome = {genome_json}",
        f"  metrics = {metrics_json}",
        f"  recent (gen, fitness) = {recent_json}  (lower fitness is better)",
        "",
        "PROPOSE exactly 3 candidate strategies to try next. Each is a JSON object "
        "with these keys:",
        '  method: "ilp" (exact, fewer clusters) or "greedy" (fast, 0 leakage)',
        "  C_max: integer 3..9 (max tasks per cluster/day)",
        "  advance_limit: integer 1..6 (max days a task may be advanced)",
        "  safety_margin: integer 1..6 (days kept above the 3.0 bar alarm)",
        "  advance_prefer: float -2..2 (negative=prefer delay/less leakage, positive=prefer advance)",
        "  w_cost, w_leak, w_reliability: the objective weights (reliability weight 1..20)",
        "Reason about the trade-off: fewer clusters cut deployment cost but delaying a "
        "task increases SF6 leakage; never allow a reliability violation. Make the 3 "
        "candidates meaningfully DIFFERENT from each other. "
        "Return ONLY a JSON array of 3 objects, no prose, no markdown.",
    ]
    return "\n".join(lines)


def propose(gen: int, best: dict, history: list, n: int = 3) -> list[Strategy]:
    """Ask the LLM to propose `n` candidate strategies; return Strategy objects."""
    out = _llm_call(_build_prompt(gen, best, history))
    genomes = _parse(out)
    if not genomes:
        return []
    strategies = []
    for g in genomes[:n]:
        cg = _clamp(g)
        s = Strategy(**cg)
        strategies.append(s)
    return strategies