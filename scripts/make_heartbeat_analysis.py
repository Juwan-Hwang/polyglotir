#!/usr/bin/env python
"""Phase 3 — Heartbeat N-value experiment: analysis & figures.

Per spec §4 Phase 3: "心跳 N 值 vs 长会话累积错误传播找拐点"
Per spec §5: "鲁棒性（L1~L4 扰动曲线 + 上下文干扰识别 + 多轮累积错误传播）"

Figures generated:
  1. fig_heartbeat_passrate_by_context  — pass rate vs context depth (turns since last heartbeat)
  2. fig_heartbeat_error_propagation    — P(fail | prev fail) vs P(fail | prev pass)
  3. fig_heartbeat_n_overall            — overall pass rate by N value (bar chart)
  4. fig_heartbeat_turn_heatmap         — N × turn pass-rate heatmap
  5. fig_heartbeat_latency              — response latency vs context depth

Usage::

    python scripts/make_heartbeat_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "phase3_heartbeat"
FIGURES_DIR = ROOT / "data" / "figures"
TABLES_DIR = ROOT / "data" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

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

MODEL_SLUGS = ["deepseek-v3-2", "glm-5-2", "kimi-k2-6"]
MODEL_LABELS = {
    "deepseek-v3-2": "DeepSeek-v3.2",
    "glm-5-2": "GLM-5.2",
    "kimi-k2-6": "Kimi-K2.6",
}
MODEL_COLORS = {
    "deepseek-v3-2": "#2563eb",
    "glm-5-2": "#16a34a",
    "kimi-k2-6": "#dc2626",
}

N_VALUES = [1, 5, 10, 15, 20, 9999]
N_LABELS = {
    1: "N=1",
    5: "N=5",
   10: "N=10",
   15: "N=15",
   20: "N=20",
   9999: "N=∞",
}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}", file=sys.stderr)
    plt.close(fig)


# ── Data loading ───────────────────────────────────────────────────────

def load_all_results() -> dict[str, list[dict]]:
    """Load all heartbeat results, keyed by model slug."""
    data: dict[str, list[dict]] = {}
    for slug in MODEL_SLUGS:
        path = RAW_DIR / slug / "results.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        data[slug] = [json.loads(l) for l in lines if l]
    return data


def _sort_key(r: dict) -> tuple:
    """Sort key for chronological ordering within a session."""
    return (r["n_value"], r["seed"], r["turn"])


# ── Figure 1: Pass rate vs context depth ───────────────────────────────

def fig_passrate_by_context(data: dict[str, list[dict]]) -> None:
    """Pass rate as a function of context depth (turns since last heartbeat).

    This is the core "error accumulation" curve: does pass rate degrade
    as the model sees more accumulated context?
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for slug in MODEL_SLUGS:
        results = data[slug]
        # Group by context_turns
        by_ctx: dict[int, list[bool]] = defaultdict(list)
        for r in results:
            by_ctx[r["context_turns"]].append(r["first_pass"])

        ctx_vals = sorted(by_ctx.keys())
        pass_rates = [np.mean(by_ctx[c]) * 100 for c in ctx_vals]
        counts = [len(by_ctx[c]) for c in ctx_vals]

        ax.plot(ctx_vals, pass_rates, "o-",
                color=MODEL_COLORS[slug], label=MODEL_LABELS[slug],
                linewidth=2, markersize=6)

    ax.set_xlabel("Context Depth (turns since last heartbeat)")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Phase 3: Pass Rate Degradation vs Context Depth")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    _save(fig, "fig_heartbeat_passrate_by_context")


# ── Figure 2: Error propagation ────────────────────────────────────────

def fig_error_propagation(data: dict[str, list[dict]]) -> None:
    """Error propagation: P(fail | prev fail) vs P(fail | prev pass).

    For each consecutive pair of turns within the same session (same
    model, N, seed), we check whether the previous turn's outcome
    affects the current turn's success.

    Only pairs where both turns are within the same heartbeat cycle
    (context_turns increases by 1) are considered — otherwise the
    heartbeat reset breaks the propagation chain.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    categories = ["P(fail | prev pass)", "P(fail | prev fail)"]
    x = np.arange(len(categories))
    width = 0.25

    for i, slug in enumerate(MODEL_SLUGS):
        results = data[slug]
        # Sort by session
        results_sorted = sorted(results, key=_sort_key)

        # Group by session
        sessions: dict[str, list[dict]] = defaultdict(list)
        for r in results_sorted:
            key = f'{r["n_value"]}|{r["seed"]}'
            sessions[key].append(r)

        # Count transitions within heartbeat cycles
        fail_after_pass = 0
        total_after_pass = 0
        fail_after_fail = 0
        total_after_fail = 0

        for sess_key, turns in sessions.items():
            turns.sort(key=lambda t: t["turn"])
            for j in range(1, len(turns)):
                prev = turns[j - 1]
                curr = turns[j]
                # Only count if within same heartbeat cycle
                # (context_turns should increase by 1)
                if curr["context_turns"] != prev["context_turns"] + 1:
                    continue
                if prev["first_pass"]:
                    total_after_pass += 1
                    if not curr["first_pass"]:
                        fail_after_pass += 1
                else:
                    total_after_fail += 1
                    if not curr["first_pass"]:
                        fail_after_fail += 1

        p_fail_given_pass = fail_after_pass / total_after_pass * 100 if total_after_pass else 0
        p_fail_given_fail = fail_after_fail / total_after_fail * 100 if total_after_fail else 0

        bars = ax.bar(x + i * width, [p_fail_given_pass, p_fail_given_fail],
                      width, color=MODEL_COLORS[slug], label=MODEL_LABELS[slug],
                      edgecolor="white")

        for bar, val, n in zip(bars, [p_fail_given_pass, p_fail_given_fail],
                                [total_after_pass, total_after_fail]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%\n(n={n})", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories)
    ax.set_ylabel("P(fail) (%)")
    ax.set_title("Phase 3: Error Propagation in Multi-Turn Sessions\n(within heartbeat cycles only)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, "fig_heartbeat_error_propagation")


# ── Figure 3: Overall pass rate by N ───────────────────────────────────

def fig_n_overall(data: dict[str, list[dict]]) -> None:
    """Overall pass rate by N value — bar chart grouped by model."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(N_VALUES))
    width = 0.22

    for i, slug in enumerate(MODEL_SLUGS):
        results = data[slug]
        rates = []
        for n in N_VALUES:
            subset = [r for r in results if r["n_value"] == n]
            rate = np.mean([r["first_pass"] for r in subset]) * 100
            rates.append(rate)

        bars = ax.bar(x + i * width, rates, width,
                      color=MODEL_COLORS[slug], label=MODEL_LABELS[slug],
                      edgecolor="white")

        # Put label INSIDE the bar (white text) to avoid horizontal overlap
        for bar, val in zip(bars, rates):
            y_pos = bar.get_height() - 3 if bar.get_height() > 15 else bar.get_height() + 1
            y_va = "top" if bar.get_height() > 15 else "bottom"
            txt_color = "white" if bar.get_height() > 15 else "black"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.1f}%", ha="center", va=y_va,
                    fontsize=7, fontweight="bold", color=txt_color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([N_LABELS[n] for n in N_VALUES], fontsize=9, rotation=0)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Phase 3: Overall Pass Rate by Heartbeat N Value")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, "fig_heartbeat_n_overall")


# ── Figure 4: N × turn heatmap ─────────────────────────────────────────

def fig_turn_heatmap(data: dict[str, list[dict]]) -> None:
    """N × turn pass-rate heatmap, one per model."""
    n_models = len(MODEL_SLUGS)
    # Use 1 row × 3 cols, taller figure for breathing room
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5),
                             sharey=True)
    if n_models == 1:
        axes = np.array([axes])

    turns = list(range(15))

    for ax_idx, slug in enumerate(MODEL_SLUGS):
        ax = axes.flat[ax_idx]
        results = data[slug]
        matrix = np.zeros((len(N_VALUES), len(turns)))

        for i, n in enumerate(N_VALUES):
            for j, t in enumerate(turns):
                subset = [r for r in results
                          if r["n_value"] == n and r["turn"] == t]
                matrix[i, j] = np.mean([r["first_pass"] for r in subset]) * 100

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "silp", ["#fef2f2", "#fca5a5", "#fbbf24", "#86efac", "#22c55e"]
        )
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        for i in range(len(N_VALUES)):
            for j in range(len(turns)):
                val = matrix[i, j]
                color = "white" if val > 70 or val < 20 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=color)

        ax.set_xticks(range(len(turns)))
        ax.set_xticklabels(turns, fontsize=7)
        ax.set_xlabel("Turn")
        if ax_idx == 0:
            ax.set_yticks(range(len(N_VALUES)))
            ax.set_yticklabels([N_LABELS[n] for n in N_VALUES], fontsize=8)
        ax.set_title(MODEL_LABELS[slug])

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Pass Rate (%)")
    fig.suptitle("Phase 3: Per-Turn Pass Rate Heatmap (N × Turn)",
                 fontsize=12, y=1.01)

    _save(fig, "fig_heartbeat_turn_heatmap")


# ── Figure 5: Latency vs context depth ─────────────────────────────────

def fig_latency_by_context(data: dict[str, list[dict]]) -> None:
    """Response latency as a function of context depth.

    While pass rate doesn't degrade, accumulated context increases
    prompt length — does this slow down responses?
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for slug in MODEL_SLUGS:
        results = data[slug]
        by_ctx: dict[int, list[float]] = defaultdict(list)
        for r in results:
            by_ctx[r["context_turns"]].append(r["elapsed"])

        ctx_vals = sorted(by_ctx.keys())
        medians = [np.median(by_ctx[c]) for c in ctx_vals]
        q25 = [np.percentile(by_ctx[c], 25) for c in ctx_vals]
        q75 = [np.percentile(by_ctx[c], 75) for c in ctx_vals]

        ax.plot(ctx_vals, medians, "o-",
                color=MODEL_COLORS[slug], label=MODEL_LABELS[slug],
                linewidth=2, markersize=5)
        ax.fill_between(ctx_vals, q25, q75, alpha=0.15,
                        color=MODEL_COLORS[slug])

    ax.set_xlabel("Context Depth (turns since last heartbeat)")
    ax.set_ylabel("Response Latency (s)")
    ax.set_title("Phase 3: Response Latency vs Context Depth\n(median + IQR band)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    _save(fig, "fig_heartbeat_latency")


# ── Statistical test: Spearman trend for context depth ────────────────

def stat_context_trend(data: dict[str, list[dict]]) -> None:
    """Spearman rank correlation: context depth vs pass rate / latency."""
    from scipy.stats import spearmanr

    print("\n" + "=" * 60, file=sys.stderr)
    print("Context Depth Trend Analysis (Spearman ρ)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for slug in MODEL_SLUGS:
        results = data[slug]

        # Pass rate trend
        by_ctx: dict[int, list[bool]] = defaultdict(list)
        by_ctx_lat: dict[int, list[float]] = defaultdict(list)
        for r in results:
            by_ctx[r["context_turns"]].append(r["first_pass"])
            by_ctx_lat[r["context_turns"]].append(r["elapsed"])

        ctxs = sorted(by_ctx.keys())
        pass_rates = [np.mean(by_ctx[c]) for c in ctxs]
        latencies = [np.median(by_ctx_lat[c]) for c in ctxs]

        rho_pass, p_pass = spearmanr(ctxs, pass_rates)
        rho_lat, p_lat = spearmanr(ctxs, latencies)

        print(f"\n  {MODEL_LABELS[slug]}:", file=sys.stderr)
        print(f"    Pass rate:  ρ={rho_pass:+.3f}, p={p_pass:.4f} "
              f"{'*' if p_pass < 0.05 else 'n.s.'}", file=sys.stderr)
        print(f"    Latency:    ρ={rho_lat:+.3f}, p={p_lat:.4f} "
              f"{'*' if p_lat < 0.05 else 'n.s.'}", file=sys.stderr)


# ── Summary table ──────────────────────────────────────────────────────

def make_summary_table(data: dict[str, list[dict]]) -> None:
    """Write a summary CSV table."""
    import csv

    path = TABLES_DIR / "phase3_heartbeat_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_value", "total", "pass", "fail",
                     "pass_rate_pct", "mean_context_turns"])

        for slug in MODEL_SLUGS:
            for n in N_VALUES:
                subset = [r for r in data[slug] if r["n_value"] == n]
                total = len(subset)
                passes = sum(1 for r in subset if r["first_pass"])
                fails = total - passes
                rate = passes / total * 100 if total else 0
                mean_ctx = np.mean([r["context_turns"] for r in subset])
                w.writerow([slug, n, total, passes, fails,
                            f"{rate:.1f}", f"{mean_ctx:.1f}"])

    print(f"  saved {path}", file=sys.stderr)


# ── Statistical test: Fisher exact for error propagation ──────────────

def stat_error_propagation(data: dict[str, list[dict]]) -> None:
    """Fisher's exact test for error propagation significance."""
    from scipy.stats import fisher_exact

    print("\n" + "=" * 60, file=sys.stderr)
    print("Error Propagation Analysis (Fisher's exact test)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for slug in MODEL_SLUGS:
        results = data[slug]
        results_sorted = sorted(results, key=_sort_key)
        sessions: dict[str, list[dict]] = defaultdict(list)
        for r in results_sorted:
            key = f'{r["n_value"]}|{r["seed"]}'
            sessions[key].append(r)

        # 2×2 contingency: [pass→pass, pass→fail, fail→pass, fail→fail]
        pp = pf = fp = ff = 0
        for sess_key, turns in sessions.items():
            turns.sort(key=lambda t: t["turn"])
            for j in range(1, len(turns)):
                prev = turns[j - 1]
                curr = turns[j]
                if curr["context_turns"] != prev["context_turns"] + 1:
                    continue
                if prev["first_pass"] and curr["first_pass"]:
                    pp += 1
                elif prev["first_pass"] and not curr["first_pass"]:
                    pf += 1
                elif not prev["first_pass"] and curr["first_pass"]:
                    fp += 1
                else:
                    ff += 1

        # Fisher exact test on [[pp, pf], [fp, ff]]
        table = [[pp, pf], [fp, ff]]
        odds, p_val = fisher_exact(table, alternative="greater")

        total_pass = pp + pf
        total_fail = fp + ff
        p_fail_given_pass = pf / total_pass * 100 if total_pass else 0
        p_fail_given_fail = ff / total_fail * 100 if total_fail else 0

        print(f"\n  {MODEL_LABELS[slug]}:", file=sys.stderr)
        print(f"    P(fail | prev pass) = {p_fail_given_pass:.1f}%  (n={total_pass})",
              file=sys.stderr)
        print(f"    P(fail | prev fail) = {p_fail_given_fail:.1f}%  (n={total_fail})",
              file=sys.stderr)
        print(f"    Contingency: pass→pass={pp}, pass→fail={pf}, "
              f"fail→pass={fp}, fail→fail={ff}", file=sys.stderr)
        print(f"    Fisher exact: odds={odds:.3f}, p={p_val:.4f}",
              file=sys.stderr)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"    Significance: {sig}", file=sys.stderr)


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...", file=sys.stderr)
    data = load_all_results()
    total = sum(len(v) for v in data.values())
    print(f"  {total} results across {len(data)} models", file=sys.stderr)

    print("\n--- Figure 1: Pass rate vs context depth ---", file=sys.stderr)
    fig_passrate_by_context(data)

    print("\n--- Figure 2: Error propagation ---", file=sys.stderr)
    fig_error_propagation(data)

    print("\n--- Figure 3: Overall pass rate by N ---", file=sys.stderr)
    fig_n_overall(data)

    print("\n--- Figure 4: N × turn heatmap ---", file=sys.stderr)
    fig_turn_heatmap(data)

    print("\n--- Figure 5: Latency vs context depth ---", file=sys.stderr)
    fig_latency_by_context(data)

    print("\n--- Summary table ---", file=sys.stderr)
    make_summary_table(data)

    print("\n--- Statistical tests ---", file=sys.stderr)
    stat_error_propagation(data)
    stat_context_trend(data)

    print(f"\nAll figures saved to {FIGURES_DIR}", file=sys.stderr)
    print(f"Summary table saved to {TABLES_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
