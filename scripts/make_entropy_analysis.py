#!/usr/bin/env python
"""Phase 3 — Entropy analysis: Shannon entropy of each frontend's encoded output.

Per spec §4 Phase 3: "熵值警戒线"

Computes per-frontend:
  - Shannon entropy (bits/char) — measures information density
  - Token efficiency — success rate per bit of entropy
  - Entropy-to-success ratio — how much entropy is "wasted"

The "entropy warning line" is the entropy level beyond which success rate
drops below 80% — frontends crossing this line sacrifice semantic fidelity
for information density.

Uses existing Phase 2 encoded data. No API calls needed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw" / "phase2"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase3_tradeoff"
FIGURES_DIR = ROOT / "data" / "figures"

FRONTENDS = ["code", "json", "natural", "nl_json", "llmlingua2"]
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6", "longcat-2.0", "minimax-m2.7"]


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy in bits/character."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _load_results() -> list[dict]:
    all_results: list[dict] = []
    for model in MODELS:
        slug = model.replace(".", "-").replace("/", "-")
        jsonl = RAW_DIR / slug / "results.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                all_results.append(json.loads(line))
    return all_results


def generate_entropy_analysis() -> None:
    results = _load_results()
    if not results:
        print("Error: no results found", file=sys.stderr)
        sys.exit(1)

    # Per-frontend: collect encoded strings, compute entropy, success rate
    fe_data: dict[str, dict] = {fe: {
        "entropies": [], "chars": [], "passes": 0, "total": 0,
        "per_model": {m: {"pass": 0, "total": 0} for m in MODELS},
    } for fe in FRONTENDS}

    for r in results:
        fe = r.get("frontend", "")
        enc = r.get("encoded", "")
        if fe not in fe_data or not enc:
            continue
        fe_data[fe]["entropies"].append(_shannon_entropy(enc))
        fe_data[fe]["chars"].append(len(enc))
        fe_data[fe]["total"] += 1
        if r.get("first_pass"):
            fe_data[fe]["passes"] += 1
        model = r.get("model", "")
        if model in fe_data[fe]["per_model"]:
            fe_data[fe]["per_model"][model]["total"] += 1
            if r.get("first_pass"):
                fe_data[fe]["per_model"][model]["pass"] += 1

    # Compute summary stats
    stats: dict[str, dict] = {}
    for fe in FRONTENDS:
        d = fe_data[fe]
        if not d["entropies"]:
            continue
        avg_entropy = sum(d["entropies"]) / len(d["entropies"])
        avg_chars = sum(d["chars"]) / len(d["chars"])
        success_rate = d["passes"] / d["total"] if d["total"] else 0.0
        total_bits = avg_entropy * avg_chars  # total information bits per encoded string

        # Cross-model variance
        model_rates = []
        for m in MODELS:
            md = d["per_model"][m]
            if md["total"]:
                model_rates.append(md["pass"] / md["total"])
        cross_std = (sum((r - success_rate) ** 2 for r in model_rates) / len(model_rates)) ** 0.5 if model_rates else 0.0

        stats[fe] = {
            "avg_entropy_bits_per_char": avg_entropy,
            "avg_chars": avg_chars,
            "total_info_bits": total_bits,
            "success_rate": success_rate,
            "cross_model_std": cross_std,
            "n": d["total"],
        }

    # Find entropy warning line: entropy level where success drops below 80%
    # Sort frontends by entropy, find the inflection point
    sorted_by_entropy = sorted(stats.items(), key=lambda x: x[1]["avg_entropy_bits_per_char"])

    # ── Write CSV ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "entropy_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frontend", "entropy_bits_per_char", "avg_chars",
            "total_info_bits", "success_rate", "cross_model_std",
        ] + MODELS)
        for fe in FRONTENDS:
            if fe not in stats:
                continue
            s = stats[fe]
            model_cols = []
            for m in MODELS:
                md = fe_data[fe]["per_model"][m]
                rate = md["pass"] / md["total"] * 100 if md["total"] else 0
                model_cols.append(f"{rate:.1f}%")
            writer.writerow([
                fe,
                f"{s['avg_entropy_bits_per_char']:.4f}",
                f"{s['avg_chars']:.1f}",
                f"{s['total_info_bits']:.1f}",
                f"{s['success_rate']*100:.1f}%",
                f"{s['cross_model_std']*100:.1f}",
            ] + model_cols)
    print(f"  CSV: {csv_path}", file=sys.stderr)

    # ── Print summary ────────────────────────────────────────────
    print(f"\n{'='*90}", file=sys.stderr)
    print(f"Entropy Analysis: Information Density vs Semantic Fidelity", file=sys.stderr)
    print(f"{'='*90}", file=sys.stderr)
    print(
        f"{'Frontend':<14} {'H (bits/char)':>14} {'Avg chars':>10}"
        f" {'Total bits':>12} {'Success':>8} {'StdDev':>8}",
        file=sys.stderr,
    )
    print(f"{'-'*14} {'-'*14} {'-'*10} {'-'*12} {'-'*8} {'-'*8}", file=sys.stderr)
    for fe in FRONTENDS:
        if fe not in stats:
            continue
        s = stats[fe]
        print(
            f"{fe:<14} {s['avg_entropy_bits_per_char']:>14.4f} {s['avg_chars']:>10.1f}"
            f" {s['total_info_bits']:>12.1f} {s['success_rate']*100:>7.1f}%"
            f" {s['cross_model_std']*100:>7.1f}%",
            file=sys.stderr,
        )

    # Entropy warning line
    print(f"\n  Entropy ordering (low → high):", file=sys.stderr)
    for fe, s in sorted_by_entropy:
        marker = " ← WARNING (<80%)" if s["success_rate"] < 0.80 else ""
        print(
            f"    {fe:<14} H={s['avg_entropy_bits_per_char']:.4f}"
            f"  success={s['success_rate']*100:.1f}%{marker}",
            file=sys.stderr,
        )

    # Total bits vs success
    print(f"\n  Total information bits vs success:", file=sys.stderr)
    for fe, s in sorted(stats.items(), key=lambda x: x[1]["total_info_bits"]):
        print(
            f"    {fe:<14} bits={s['total_info_bits']:>8.1f}"
            f"  success={s['success_rate']*100:.1f}%",
            file=sys.stderr,
        )
    print(f"{'='*90}", file=sys.stderr)

    # ── Generate figure ──────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for fe in FRONTENDS:
            if fe not in stats:
                continue
            s = stats[fe]
            x = s["avg_entropy_bits_per_char"]
            y = s["success_rate"] * 100
            ax.scatter(x, y, s=150, zorder=5)
            ax.annotate(
                fe, (x, y), textcoords="offset points",
                xytext=(10, 5), fontsize=11, fontweight="bold",
            )

        ax.set_xlabel("Shannon Entropy (bits/char)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title("Entropy vs Semantic Fidelity", fontsize=14)
        ax.axhline(y=80, color="red", linestyle="--", alpha=0.4, label="80% warning line")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2, 5)

        fig.tight_layout()
        fig_path = FIGURES_DIR / "entropy_curve.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  Figure: {fig_path}", file=sys.stderr)

        # Also: total bits vs success
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        for fe in FRONTENDS:
            if fe not in stats:
                continue
            s = stats[fe]
            x = s["total_info_bits"]
            y = s["success_rate"] * 100
            ax2.scatter(x, y, s=150, zorder=5)
            ax2.annotate(
                fe, (x, y), textcoords="offset points",
                xytext=(10, 5), fontsize=11, fontweight="bold",
            )

        ax2.set_xlabel("Total Information Bits (entropy × chars)", fontsize=12)
        ax2.set_ylabel("Success Rate (%)", fontsize=12)
        ax2.set_title("Total Information vs Semantic Fidelity", fontsize=14)
        ax2.axhline(y=80, color="red", linestyle="--", alpha=0.4, label="80% warning line")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2_path = FIGURES_DIR / "entropy_total_bits.png"
        fig2.savefig(fig2_path, dpi=150)
        print(f"  Figure: {fig2_path}", file=sys.stderr)

    except ImportError:
        print("  [skip] matplotlib not available", file=sys.stderr)


if __name__ == "__main__":
    generate_entropy_analysis()
