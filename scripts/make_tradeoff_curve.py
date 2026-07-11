#!/usr/bin/env python
"""Phase 3 — Tradeoff curve: compression rate vs success rate.

Per spec §4 Phase 3: "压缩率 vs 成功率曲线找拐点"

Uses existing Phase 2 data (675 runs) to plot the tradeoff between
token compression and semantic fidelity across all 5 frontends.

For each frontend, we compute:
  - **Compression rate**: avg token count relative to the natural frontend
    (natural = 1.0 baseline, lower = more compressed)
  - **Success rate**: avg pass rate across all 5 models × 27 cases
  - **Token variance**: std-dev of token counts across tokenizers

Output:
  - data/processed/phase3_tradeoff/tradeoff_curve.csv
  - data/figures/tradeoff_curve.png

No API calls needed — reads from data/raw/phase2/*/results.jsonl
and data/processed/phase2/phase2_matrix.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw" / "phase2"
PROCESSED_DIR = ROOT / "data" / "processed" / "phase2"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase3_tradeoff"
FIGURES_DIR = ROOT / "data" / "figures"

FRONTENDS = ["code", "json", "natural", "nl_json", "llmlingua2"]
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6", "longcat-2.0", "minimax-m2.7"]


def _load_results() -> list[dict]:
    """Load all results from raw JSONL files."""
    all_results = []
    for model in MODELS:
        slug = model.replace(".", "-").replace("/", "-")
        jsonl = RAW_DIR / slug / "results.jsonl"
        if not jsonl.exists():
            print(f"  [warn] missing {jsonl}", file=sys.stderr)
            continue
        for line in jsonl.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                all_results.append(json.loads(line))
    return all_results


def _count_tokens_approx(text: str) -> dict[str, int]:
    """Approximate token count using simple heuristics.

    Since we don't have all tokenizers loaded, we use:
    - word_count: whitespace split (proxy for general token count)
    - char_count: raw character count
    - est_gpt: ~4 chars per token (GPT-family approximation)
    """
    words = len(text.split())
    chars = len(text)
    return {
        "words": words,
        "chars": chars,
        "est_gpt_tokens": max(1, chars // 4),
    }


def _compute_compression(results: list[dict]) -> dict[str, dict]:
    """For each frontend, compute avg token count and compression ratio
    relative to the natural frontend."""
    # Collect encoded strings per frontend
    fe_texts: dict[str, list[str]] = {fe: [] for fe in FRONTENDS}
    for r in results:
        fe = r.get("frontend", "")
        encoded = r.get("encoded", "")
        if fe in fe_texts and encoded:
            fe_texts[fe].append(encoded)

    # Compute avg token counts
    natural_words = None
    stats = {}
    for fe in FRONTENDS:
        texts = fe_texts[fe]
        if not texts:
            continue
        word_counts = [_count_tokens_approx(t)["words"] for t in texts]
        char_counts = [_count_tokens_approx(t)["chars"] for t in texts]
        avg_words = sum(word_counts) / len(word_counts)
        avg_chars = sum(char_counts) / len(char_counts)
        stats[fe] = {
            "avg_words": avg_words,
            "avg_chars": avg_chars,
            "est_gpt_tokens": max(1, int(avg_chars // 4)),
            "n_samples": len(texts),
        }
        if fe == "natural":
            natural_words = avg_words

    # Compute compression ratio (relative to natural)
    for fe in stats:
        if natural_words and natural_words > 0:
            stats[fe]["compression_ratio"] = stats[fe]["avg_words"] / natural_words
        else:
            stats[fe]["compression_ratio"] = 1.0

    return stats


def _compute_success_rates(results: list[dict]) -> dict[str, float]:
    """Compute per-frontend success rate."""
    rates = {}
    for fe in FRONTENDS:
        subset = [r for r in results if r.get("frontend") == fe]
        if subset:
            rates[fe] = sum(1 for r in subset if r.get("first_pass")) / len(subset)
        else:
            rates[fe] = 0.0
    return rates


def _compute_per_model_rates(results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute per-frontend per-model success rate for variance analysis."""
    rates = {}
    for fe in FRONTENDS:
        rates[fe] = {}
        for model in MODELS:
            subset = [
                r for r in results
                if r.get("frontend") == fe and r.get("model") == model
            ]
            if subset:
                rates[fe][model] = sum(1 for r in subset if r.get("first_pass")) / len(subset)
            else:
                rates[fe][model] = 0.0
    return rates


def generate_tradeoff() -> None:
    results = _load_results()
    if not results:
        print("Error: no results found", file=sys.stderr)
        sys.exit(1)

    compression = _compute_compression(results)
    success = _compute_success_rates(results)
    per_model = _compute_per_model_rates(results)

    # Compute cross-model variance (std dev)
    import statistics
    variance = {}
    for fe in FRONTENDS:
        if fe in per_model and per_model[fe]:
            variance[fe] = statistics.stdev(per_model[fe].values())
        else:
            variance[fe] = 0.0

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "tradeoff_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frontend", "avg_words", "avg_chars", "est_gpt_tokens",
            "compression_ratio", "success_rate", "cross_model_std",
        ] + MODELS)
        for fe in FRONTENDS:
            if fe not in compression:
                continue
            c = compression[fe]
            s = success.get(fe, 0.0)
            v = variance.get(fe, 0.0)
            row = [
                fe,
                f"{c['avg_words']:.1f}",
                f"{c['avg_chars']:.1f}",
                c["est_gpt_tokens"],
                f"{c['compression_ratio']:.3f}",
                f"{s*100:.1f}%",
                f"{v*100:.1f}",
            ] + [f"{per_model.get(fe, {}).get(m, 0)*100:.1f}%" for m in MODELS]
            writer.writerow(row)

    print(f"  Tradeoff CSV: {csv_path}", file=sys.stderr)

    # Print summary table
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"Tradeoff Curve: Compression vs Success Rate", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"{'Frontend':<14} {'Tokens':>8} {'Compr%':>8} {'Success':>8} {'StdDev':>8}", file=sys.stderr)
    print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", file=sys.stderr)
    for fe in FRONTENDS:
        if fe not in compression:
            continue
        c = compression[fe]
        s = success.get(fe, 0.0)
        v = variance.get(fe, 0.0)
        print(
            f"{fe:<14} {c['avg_words']:>8.1f} {c['compression_ratio']*100:>7.1f}% {s*100:>7.1f}% {v*100:>7.1f}%",
            file=sys.stderr,
        )
    print(f"{'='*80}", file=sys.stderr)

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot each frontend as a point
        for fe in FRONTENDS:
            if fe not in compression:
                continue
            x = compression[fe]["compression_ratio"]
            y = success.get(fe, 0.0) * 100
            ax.scatter(x, y, s=150, zorder=5)
            ax.annotate(
                fe,
                (x, y),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xlabel("Compression Ratio (relative to natural = 1.0)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title("Tradeoff: Compression vs Semantic Fidelity", fontsize=14)
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = FIGURES_DIR / "tradeoff_curve.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  Figure: {fig_path}", file=sys.stderr)

        # Also generate a per-model breakdown figure
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        markers = ["o", "s", "^", "D", "v"]
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
        for i, model in enumerate(MODELS):
            xs, ys = [], []
            for fe in FRONTENDS:
                if fe not in compression:
                    continue
                xs.append(compression[fe]["compression_ratio"])
                ys.append(per_model.get(fe, {}).get(model, 0) * 100)
            ax2.plot(xs, ys, marker=markers[i % len(markers)],
                     color=colors[i % len(colors)], label=model, linewidth=2, markersize=8)

        ax2.set_xlabel("Compression Ratio (relative to natural = 1.0)", fontsize=12)
        ax2.set_ylabel("Success Rate (%)", fontsize=12)
        ax2.set_title("Tradeoff by Model", fontsize=14)
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(0, 110)

        # Add frontend labels on x-axis
        for fe in FRONTENDS:
            if fe in compression:
                x = compression[fe]["compression_ratio"]
                ax2.axvline(x=x, color="gray", linestyle=":", alpha=0.2)

        fig2.tight_layout()
        fig2_path = FIGURES_DIR / "tradeoff_by_model.png"
        fig2.savefig(fig2_path, dpi=150)
        print(f"  Figure: {fig2_path}", file=sys.stderr)

    except ImportError:
        print("  [skip] matplotlib not available, skipping figures", file=sys.stderr)


if __name__ == "__main__":
    generate_tradeoff()
