"""
Generate plots for the final report:
  (a) self-evolution convergence (fitness & cost-reduction over generations)
  (b) comparison bar chart (clusters & cost reduction across 3 methods)
  (c) advance/delay profile comparison
Reads results/evaluation.json (produced by evaluate/compare.py).
"""
from __future__ import annotations

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/ubadmin/projects/AgentOpt/results"


def load():
    with open(f"{OUT}/evaluation.json") as f:
        return json.load(f)


def plot_evolution(ev):
    hist = ev.get("evolution_history") or ev.get("history") or []
    gens = [h["gen"] for h in hist]
    red = [h["best_metrics"]["cost_reduction"] for h in hist]
    clu = [h["best_metrics"]["n_clusters"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(gens, red, "o-", color="#1f77b4", lw=2)
    ax1.set_title("Self-evolution: cost reduction by generation")
    ax1.set_xlabel("generation"); ax1.set_ylabel("cost reduction")
    ax1.set_yticklabels([f"{r*100:.0f}%" for r in ax1.get_yticks()])
    ax1.grid(True, alpha=0.3)
    ax2.bar(gens, clu, color="#ff7f0e")
    ax2.set_title("Self-evolution: # clusters (deployments) by generation")
    ax2.set_xlabel("generation"); ax2.set_ylabel("clusters")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUT}/evolution.png", dpi=130)
    plt.close()


def plot_comparison(ev):
    rows = ev["baselines"]
    methods = list(rows.keys())
    clusters = [rows[m]["n_clusters"] for m in methods]
    red = [rows[m]["cost_reduction"] for m in methods]
    viol = [rows[m]["n_violations"] for m in methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(x, clusters, color="#2ca02c")
    axes[0].set_title("# clusters (deployments)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=15, ha="right")
    for i, c in enumerate(clusters):
        axes[0].text(i, c + 0.3, str(c), ha="center")
    axes[1].bar(x, [r * 100 for r in red], color="#1f77b4")
    axes[1].set_title("cost reduction (%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=15, ha="right")
    axes[1].set_ylim(0, 100)
    for i, r in enumerate(red):
        axes[1].text(i, r * 100 + 2, f"{r*100:.1f}%", ha="center")
    axes[2].bar(x, viol, color="#d62728")
    axes[2].set_title("reliability violations")
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods, rotation=15, ha="right")
    for i, v in enumerate(viol):
        axes[2].text(i, v + 0.1, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison.png", dpi=130)
    plt.close()


def main():
    ev = load()
    plot_evolution(ev)
    plot_comparison(ev)
    print("saved results/evolution.png and results/comparison.png")


if __name__ == "__main__":
    main()