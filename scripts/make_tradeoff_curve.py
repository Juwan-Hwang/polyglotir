#!/usr/bin/env python
"""Phase 3 — Tradeoff curve: compression rate vs success rate.

Per spec §4 Phase 3: "压缩率 vs 成功率曲线找拐点"

Uses existing Phase 2 data (675 runs) to plot the tradeoff between
token compression and semantic fidelity across all 5 frontends.

Compression is measured by **character count** (not word count, since
JSON frontends produce spaceless single-line output that breaks
whitespace-based tokenisation).  The natural frontend serves as the
1.0 baseline.

Output:
  - data/processed/phase3_tradeoff/tradeoff_curve.csv
  - data/figures/tradeoff_curve.png
  - data/figures/tradeoff_by_model.png

No API calls needed — reads from data/raw/phase2/*/results.jsonl.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw" / "phase2"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase3_tradeoff"
FIGURES_DIR = ROOT / "data" / "figures"

FRONTENDS = ["code", "json", "natural", "nl_json", "llmlingua2"]
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6", "longcat-2.0", "minimax-m2.7"]


def _load_results() -> list[dict]:
    """Load all results from raw JSONL files."""
    all_results: list[dict] = []
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


def _compute_compression(results: list[dict]) -> dict[str, dict]:
    """Char-based compression ratio relative to the natural frontend.

    JSON frontends produce spaceless single-line output, so word-count
    heuristics are unreliable.  Character count is a stable proxy that
    correlates linearly with token count across tokenizer families.
    """
    fe_chars: dict[str, list[int]] = {fe: [] for fe in FRONTENDS}
    for r in results:
        fe = r.get("frontend", "")
        enc = r.get("encoded", "")
        if fe in fe_chars and enc:
            fe_chars[fe].append(len(enc))

    natural_avg = (
        sum(fe_chars["natural"]) / len(fe_chars["natural"])
        if fe_chars["natural"] else 1.0
    )

    stats: dict[str, dict] = {}
    for fe in FRONTENDS:
        chars = fe_chars[fe]
        if not chars:
            continue
        avg_chars = sum(chars) / len(chars)
        est_tokens = max(1, avg_chars / 4.0)  # GPT-family ~4 chars/token
        stats[fe] = {
            "avg_chars": avg_chars,
            "est_gpt_tokens": est_tokens,
            "compression_ratio": avg_chars / natural_avg,
            "n_samples": len(chars),
        }
    return stats


def _compute_success_rates(results: list[dict]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for fe in FRONTENDS:
        subset = [r for r in results if r.get("frontend") == fe]
        rates[fe] = sum(1 for r in subset if r.get("first_pass")) / len(subset) if subset else 0.0
    return rates


def _compute_per_model_rates(results: list[dict]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    for fe in FRONTENDS:
        rates[fe] = {}
        for model in MODELS:
            subset = [r for r in results if r.get("frontend") == fe and r.get("model") == model]
            rates[fe][model] = sum(1 for r in subset if r.get("first_pass")) / len(subset) if subset else 0.0
    return rates


def _mcnemar(results: list[dict], fe_a: str, fe_b: str) -> dict:
    """McNemar's test on paired (case, model) binary outcomes."""
    lookup: dict[tuple[str, str, str], bool] = {}
    for r in results:
        lookup[(r["case_id"], r["frontend"], r["model"])] = r.get("first_pass", False)

    keys = sorted({(r["case_id"], r["model"]) for r in results})
    b = c = 0  # b: a-pass b-fail, c: a-fail b-pass
    for case_id, model in keys:
        pa = lookup.get((case_id, fe_a, model))
        pb = lookup.get((case_id, fe_b, model))
        if pa is None or pb is None:
            continue
        if pa and not pb:
            b += 1
        elif not pa and pb:
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    # Approximate p-value for chi-squared with 1 df: p = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p": p}


def generate_tradeoff() -> None:
    results = _load_results()
    if not results:
        print("Error: no results found", file=sys.stderr)
        sys.exit(1)

    compression = _compute_compression(results)
    success = _compute_success_rates(results)
    per_model = _compute_per_model_rates(results)

    variance: dict[str, float] = {}
    for fe in FRONTENDS:
        vals = list(per_model.get(fe, {}).values())
        variance[fe] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    # Pairwise McNemar p-values
    mcnemar_results: dict[tuple[str, str], dict] = {}
    for i, a in enumerate(FRONTENDS):
        for b in FRONTENDS[i + 1:]:
            mcnemar_results[(a, b)] = _mcnemar(results, a, b)

    # ── Write CSV ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "tradeoff_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frontend", "avg_chars", "est_gpt_tokens",
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
                f"{c['avg_chars']:.1f}",
                f"{c['est_gpt_tokens']:.1f}",
                f"{c['compression_ratio']:.3f}",
                f"{s*100:.1f}%",
                f"{v*100:.1f}",
            ] + [f"{per_model.get(fe, {}).get(m, 0)*100:.1f}%" for m in MODELS]
            writer.writerow(row)
    print(f"  Tradeoff CSV: {csv_path}", file=sys.stderr)

    # McNemar CSV
    mc_path = OUTPUT_DIR / "mcnemar_pairwise.csv"
    with open(mc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frontend_a", "frontend_b", "rate_a", "rate_b", "b", "c", "chi2", "p_value", "significant"])
        for (a, b), mc in mcnemar_results.items():
            ra = success.get(a, 0) * 100
            rb = success.get(b, 0) * 100
            sig = "yes" if mc["p"] < 0.05 else "no"
            writer.writerow([a, b, f"{ra:.1f}%", f"{rb:.1f}%", mc["b"], mc["c"], f"{mc['chi2']:.4f}", f"{mc['p']:.6f}", sig])
    print(f"  McNemar CSV: {mc_path}", file=sys.stderr)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'='*90}", file=sys.stderr)
    print(f"Tradeoff Curve: Compression (char-based) vs Success Rate", file=sys.stderr)
    print(f"{'='*90}", file=sys.stderr)
    print(f"{'Frontend':<14} {'Chars':>7} {'EstTok':>7} {'Compr%':>8} {'Success':>8} {'StdDev':>8}", file=sys.stderr)
    print(f"{'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8}", file=sys.stderr)
    for fe in FRONTENDS:
        if fe not in compression:
            continue
        c = compression[fe]
        s = success.get(fe, 0.0)
        v = variance.get(fe, 0.0)
        print(
            f"{fe:<14} {c['avg_chars']:>7.1f} {c['est_gpt_tokens']:>7.1f}"
            f" {c['compression_ratio']*100:>7.1f}% {s*100:>7.1f}% {v*100:>7.1f}%",
            file=sys.stderr,
        )
    print(f"\n  Pairwise McNemar's tests (p < 0.05 = significant):", file=sys.stderr)
    for (a, b), mc in mcnemar_results.items():
        sig = "***" if mc["p"] < 0.001 else "**" if mc["p"] < 0.01 else "*" if mc["p"] < 0.05 else ""
        if sig:
            print(
                f"    {a:12s} vs {b:12s}: b={mc['b']} c={mc['c']} p={mc['p']:.6f} {sig}",
                file=sys.stderr,
            )
    print(f"  (non-significant pairs omitted)", file=sys.stderr)
    print(f"{'='*90}", file=sys.stderr)

    # ── Generate figures ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Main figure: compression vs success, with Pareto frontier
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        points = []
        for fe in FRONTENDS:
            if fe not in compression:
                continue
            x = compression[fe]["compression_ratio"]
            y = success.get(fe, 0.0) * 100
            points.append((x, y, fe))
            ax.scatter(x, y, s=150, zorder=5)
            ax.annotate(fe, (x, y), textcoords="offset points",
                        xytext=(10, 5), fontsize=11, fontweight="bold")

        # Draw Pareto frontier: points that are not dominated
        # (lower compression AND higher success = better)
        # A point is dominated if another has <= compression AND >= success
        pareto = []
        for x, y, fe in points:
            dominated = any(
                px <= x and py >= y and (px < x or py > y)
                for px, py, _ in points
            )
            if not dominated:
                pareto.append((x, y, fe))
        if len(pareto) >= 2:
            pareto.sort(key=lambda p: p[0])
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, "k--", alpha=0.3, linewidth=1.5, label="Pareto frontier")

        ax.set_xlabel("Compression Ratio (char-based, natural = 1.0)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title("Tradeoff: Compression vs Semantic Fidelity\n"
                     "(code vs json: p ≈ 0.98, NOT significant)", fontsize=13)
        ax.axhline(y=80, color="gray", linestyle=":", alpha=0.4, label="80% threshold")
        ax.set_xlim(-0.05, 1.2)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = FIGURES_DIR / "tradeoff_curve.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  Figure: {fig_path}", file=sys.stderr)

        # Per-model breakdown
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

        ax2.set_xlabel("Compression Ratio (char-based, natural = 1.0)", fontsize=12)
        ax2.set_ylabel("Success Rate (%)", fontsize=12)
        ax2.set_title("Tradeoff by Model", fontsize=14)
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.2)
        ax2.set_ylim(0, 110)

        fig2.tight_layout()
        fig2_path = FIGURES_DIR / "tradeoff_by_model.png"
        fig2.savefig(fig2_path, dpi=150)
        print(f"  Figure: {fig2_path}", file=sys.stderr)

    except ImportError:
        print("  [skip] matplotlib not available, skipping figures", file=sys.stderr)


if __name__ == "__main__":
    generate_tradeoff()
