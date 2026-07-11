#!/usr/bin/env python
"""Generate all paper figures from Phase 2 processed data.

Outputs to ``data/figures/`` (300 DPI PNG + PDF).

Figures generated:
  1. fig1_passrate_heatmap   — frontend × model pass-rate matrix (paper Table 1 visual)
  2. fig2_spearman_scatter   — pairwise Spearman ρ scatter / bar
  3. fig3_token_variance     — token-count variance across frontends (Phase 0)
  4. fig4_frontend_ranking   — frontend average pass-rate bar chart
  5. fig5_case_level_heatmap — case × frontend pass/fail grid

Usage::

    python scripts/make_figures.py
    python scripts/make_figures.py --only heatmap
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Frontend display order (best → worst expected)
FE_ORDER = ["code", "json", "natural", "nl_json", "llmlingua2"]
FE_LABELS = {
    "code": "Code",
    "json": "JSON",
    "natural": "Natural",
    "nl_json": "NL+JSON",
    "llmlingua2": "LLMLingua2",
}

# Frontend colors
FE_COLORS = {
    "code": "#2563eb",
    "json": "#0891b2",
    "natural": "#16a34a",
    "nl_json": "#ca8a04",
    "llmlingua2": "#dc2626",
}


def _save(fig, name: str) -> None:
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}", file=sys.stderr)
    plt.close(fig)


# ── Figure 1: Pass-rate heatmap ────────────────────────────────────────

def fig_passrate_heatmap() -> None:
    """Frontend × model pass-rate heatmap (the core Phase 2 matrix)."""
    csv_path = PROCESSED / "phase2" / "phase2_matrix.csv"
    if not csv_path.exists():
        print("  [skip] phase2_matrix.csv not found", file=sys.stderr)
        return

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    # Parse model names from header (exclude 'frontend', 'avg', 'rank')
    fields = list(rows[0].keys())
    models = [f for f in fields if f not in ("frontend", "avg", "rank")]

    # Build matrix: frontends (rows) × models (cols)
    frontends = [r["frontend"] for r in rows]
    data = np.zeros((len(frontends), len(models)))
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            val = r[m].strip()
            if val and val != "N/A":
                data[i, j] = float(val.rstrip("%")) / 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "silp", ["#fef2f2", "#fca5a5", "#86efac", "#22c55e", "#15803d"]
    )
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(frontends)):
        for j in range(len(models)):
            val = data[i, j]
            color = "white" if val > 0.6 or val < 0.15 else "black"
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(range(len(frontends)))
    ax.set_yticklabels([FE_LABELS.get(f, f) for f in frontends])
    ax.set_xlabel("Model")
    ax.set_ylabel("Frontend")
    ax.set_title("Phase 2: Pass-Rate Matrix (Frontend × Model)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pass Rate")

    _save(fig, "fig1_passrate_heatmap")


# ── Figure 2: Spearman correlation bar chart ──────────────────────────

def fig_spearman_scatter() -> None:
    """Pairwise Spearman ρ between model pairs."""
    json_path = PROCESSED / "phase2" / "phase2_spearman.json"
    if not json_path.exists():
        print("  [skip] phase2_spearman.json not found", file=sys.stderr)
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    pairs = data["pairs"]

    labels = [f"{p['model_a']}\nvs\n{p['model_b']}" for p in pairs]
    rhos = [p["spearman_rho"] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#15803d" if r >= 0.6 else "#ca8a04" if r >= 0.4 else "#dc2626" for r in rhos]
    bars = ax.barh(range(len(rhos)), rhos, color=colors, edgecolor="white", height=0.6)

    for i, (bar, rho) in enumerate(zip(bars, rhos)):
        ax.text(rho + 0.01, i, f"ρ={rho:.3f}", va="center", fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman ρ")
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="ρ=0.5")
    ax.set_title("Phase 2: Pairwise Spearman Rank Correlation (per-case, 45 points)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    _save(fig, "fig2_spearman_bar")


# ── Figure 3: Token-count variance across frontends ───────────────────

def fig_token_variance() -> None:
    """Token-count variance (range) across frontends from Phase 0."""
    csv_path = PROCESSED / "phase0" / "tokenizer_variance.csv"
    if not csv_path.exists():
        print("  [skip] tokenizer_variance.csv not found", file=sys.stderr)
        return

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    # Group by frontend, compute mean range
    fe_ranges: dict[str, list[float]] = {}
    for r in rows:
        fe = r["key"].split("|")[1]
        fe_ranges.setdefault(fe, []).append(float(r["range"]))

    frontends = [fe for fe in FE_ORDER if fe in fe_ranges]
    means = [np.mean(fe_ranges[fe]) for fe in frontends]
    stds = [np.std(fe_ranges[fe]) for fe in frontends]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [FE_COLORS.get(fe, "#888") for fe in frontends]
    bars = ax.bar(range(len(frontends)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", width=0.6)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(frontends)))
    ax.set_xticklabels([FE_LABELS.get(fe, fe) for fe in frontends])
    ax.set_ylabel("Token Count Range (max − min across tokenizers)")
    ax.set_title("Phase 0: Token-Count Variance Across Frontends\n(Lower = More Cross-Tokenizer Stable)")

    _save(fig, "fig3_token_variance")


# ── Figure 4: Frontend average pass-rate ranking ───────────────────────

def fig_frontend_ranking() -> None:
    """Frontend average pass-rate bar chart."""
    csv_path = PROCESSED / "phase2" / "phase2_matrix.csv"
    if not csv_path.exists():
        print("  [skip] phase2_matrix.csv not found", file=sys.stderr)
        return

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    frontends = [r["frontend"] for r in rows]
    avgs = [float(r["avg"].rstrip("%")) for r in rows]

    # Sort by avg descending
    order = np.argsort(avgs)[::-1]
    frontends = [frontends[i] for i in order]
    avgs = [avgs[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [FE_COLORS.get(fe, "#888") for fe in frontends]
    bars = ax.bar(range(len(frontends)), avgs, color=colors, edgecolor="white", width=0.55)

    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(frontends)))
    ax.set_xticklabels([FE_LABELS.get(fe, fe) for fe in frontends])
    ax.set_ylabel("Average Pass Rate (%)")
    ax.set_ylim(0, max(avgs) * 1.2 if avgs else 100)
    ax.set_title("Phase 2: Frontend Average Pass Rate (across 5 models)")

    _save(fig, "fig4_frontend_ranking")


# ── Figure 5: Case-level pass/fail grid ───────────────────────────────

def fig_case_level_heatmap() -> None:
    """Case × model pass/fail grid for the code frontend (detailed view)."""
    csv_path = PROCESSED / "phase2" / "phase2_case_details.csv"
    if not csv_path.exists():
        print("  [skip] phase2_case_details.csv not found", file=sys.stderr)
        return

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    # Get unique cases, models, frontends
    cases = sorted(set(r["case_id"] for r in rows))
    models = sorted(set(r["model"] for r in rows))
    frontends = [fe for fe in FE_ORDER if any(r["frontend"] == fe for r in rows)]

    # Build a 3D view: for each frontend, case × model matrix
    n_fe = len(frontends)
    n_cases = len(cases)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_fe, figsize=(3.5 * n_fe, 5), sharey=True)
    if n_fe == 1:
        axes = [axes]

    for ax_idx, fe in enumerate(frontends):
        ax = axes[ax_idx]
        matrix = np.zeros((n_cases, n_models))
        for i, case_id in enumerate(cases):
            for j, model in enumerate(models):
                for r in rows:
                    if r["case_id"] == case_id and r["model"] == model and r["frontend"] == fe:
                        # llm_verdict or judge_verdict
                        verdict = r.get("llm_verdict", "").strip()
                        if verdict == "pass":
                            matrix[i, j] = 1
                        else:
                            matrix[i, j] = 0
                        break

        cmap = mcolors.ListedColormap(["#fca5a5", "#86efac"])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        # Annotate
        for i in range(n_cases):
            for j in range(n_models):
                symbol = "✓" if matrix[i, j] else "✗"
                color = "#15803d" if matrix[i, j] else "#dc2626"
                ax.text(j, i, symbol, ha="center", va="center", fontsize=12, color=color)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        if ax_idx == 0:
            ax.set_yticks(range(n_cases))
            ax.set_yticklabels(cases, fontsize=8)
        ax.set_title(FE_LABELS.get(fe, fe), fontsize=11)

    fig.suptitle("Phase 2: Case-Level Pass/Fail Grid (✓ = pass, ✗ = fail)", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, "fig5_case_level_grid")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--only", choices=["heatmap", "spearman", "variance", "ranking", "case_grid"],
                        default=None, help="Generate only one figure")
    args = parser.parse_args()

    figures = {
        "heatmap": fig_passrate_heatmap,
        "spearman": fig_spearman_scatter,
        "variance": fig_token_variance,
        "ranking": fig_frontend_ranking,
        "case_grid": fig_case_level_heatmap,
    }

    if args.only:
        figures[args.only]()
    else:
        for name, func in figures.items():
            print(f"\n--- {name} ---", file=sys.stderr)
            func()

    print(f"\nAll figures saved to {FIGURES_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
