#!/usr/bin/env python
"""Phase 3 — Entropy analysis (v3): compression-based entropy + honest stats.

This script has been through three iterations of methodological correction:

v1 (withdrawn): monogram Shannon entropy → "H ≈ 4.2 warning line"
  Problem: monogram entropy measures character-set diversity, not information
  density. JSON's repetitive ``{}":,`` inflates diversity without adding
  information.

v2 (withdrawn): raw DEFLATE entropy + "progressive compression sweep"
  Problem: (a) 5 discrete frontend types yield ρ=-0.3, p=0.62 — no significant
  relationship can be claimed. (b) The "sweep" used random character deletion,
  which tests corruption robustness, not compression strength.

v3 (this version):
  - Entropy metric: raw DEFLATE (retained from v2 — the metric itself is sound)
  - Statistical claim: NONE. 5 points cannot establish a relationship.
    Spearman ρ and p-value are reported for transparency, with explicit
    "not significant" labelling.
  - Sweep data (if present): labelled as "character-level noise robustness"
    — a DIFFERENT experiment from compression-strength testing. Do not
    conflate the two.
  - All "warning line", "threshold", "inflection point" language removed.
  - The only valid claim: zlib entropy is a better *metric* than monogram
    entropy (it correctly identifies JSON as low-entropy, which monogram
    misses). This is a methodological observation, not a finding about
    entropy-success relationships.

If semantic compression sweep data (from run_compression_sweep.py) is
available, it is plotted separately as "semantic compression strength"
— this IS a valid compression-strength variable (stop-word removal),
comparable to llmlingua2's approach.

Uses existing Phase 2 data + optional sweep data. No API calls needed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw" / "phase2"
NOISE_SWEEP_DIR = ROOT / "data" / "raw" / "phase3_entropy_sweep"
COMP_SWEEP_DIR = ROOT / "data" / "raw" / "phase3_compression_sweep"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase3_tradeoff"
FIGURES_DIR = ROOT / "data" / "figures"

FRONTENDS = ["code", "json", "natural", "nl_json", "llmlingua2"]
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6", "longcat-2.0", "minimax-m2.7"]
SWEEP_MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6"]


# ── Entropy metrics ───────────────────────────────────────────────────


def zlib_entropy(text: str) -> float:
    """Compression-based empirical entropy in bits/char.

    Uses raw DEFLATE (LZ77 + Huffman, no zlib header/trailer) to estimate
    the true information content of *text*. This captures sequence-level
    redundancy — the standard information-theoretic approach for estimating
    natural language entropy.

    Raw DEFLATE is used instead of ``zlib.compress`` to minimise overhead
    for short texts (zlib adds ~6 bytes of header/checksum, which inflates
    entropy estimates for the 40–170 char strings in this project).

    **Limitation**: for very short texts (<100 chars), even raw DEFLATE
    cannot fully exploit LZ77 redundancy. Absolute values are inflated, but
    relative comparisons across frontends remain valid.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    co = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed = co.compress(raw) + co.flush()
    return len(compressed) * 8 / len(text)


def ngram_entropy(text: str, n: int = 2) -> float:
    """N-gram Shannon entropy in bits/char."""
    if len(text) < n:
        return 0.0
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    h = 0.0
    for count in counts.values():
        p = count / total
        h -= p * math.log2(p)
    return h / n


def monogram_entropy(text: str) -> float:
    """Character-frequency Shannon entropy in bits/char.

    **Known to overestimate** for structured formats. Retained for comparison.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    h = 0.0
    for count in counts.values():
        p = count / total
        h -= p * math.log2(p)
    return h


# ── Spearman correlation ──────────────────────────────────────────────


def _spearman_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rank correlation + approximate p-value (t-distribution).

    Returns (rho, p_value). For n < 5, p is not meaningful and returns 1.0.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    def _rank(values: list[float]) -> list[float]:
        indexed = sorted(range(len(values)), key=lambda i: -values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and values[indexed[j]] == values[indexed[i]]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k]] = avg
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1.0 - (6.0 * d_sq) / (n * (n * n - 1))

    # Approximate p-value using t-distribution
    if n < 5:
        return rho, 1.0  # too few points for meaningful p
    t_stat = rho * math.sqrt((n - 2) / max(1e-10, 1 - rho * rho))
    # Two-tailed p from t-distribution (approximation via normal for large n)
    # For small n, use the exact t-distribution CDF approximation
    p = _t_distribution_p_two_tailed(t_stat, n - 2)
    return rho, p


def _t_distribution_p_two_tailed(t: float, df: int) -> float:
    """Approximate two-tailed p-value for Student's t-distribution.

    Uses the incomplete beta function approximation.
    """
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    # Regularised incomplete beta function I_x(df/2, 1/2)
    ib = _incomplete_beta(x, df / 2.0, 0.5)
    return min(1.0, max(0.0, ib))


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Logarithm of prefactor
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) / a
    # Continued fraction (Lentz's method)
    if x < (a + 1) / (a + b + 2):
        return front * _beta_cf(x, a, b)
    return 1.0 - (math.exp(math.log1p(-x) * b + math.log(x) * a - lbeta) / b) * _beta_cf(1 - x, b, a)


def _beta_cf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 1e-15) -> float:
    """Continued fraction for incomplete beta function."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


# ── Data loading ──────────────────────────────────────────────────────


def _load_jsonl_dir(base_dir: Path, models: list[str]) -> list[dict]:
    """Load all results from a directory of per-model JSONL files."""
    results: list[dict] = []
    if not base_dir.exists():
        return results
    for model in models:
        slug = model.replace(".", "-").replace("/", "-")
        jsonl = base_dir / slug / "results.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                results.append(json.loads(line))
    return results


# ── Analysis ──────────────────────────────────────────────────────────


def _compute_stats(
    results: list[dict],
    frontend_labels: list[str],
) -> dict[str, dict]:
    """Compute per-frontend entropy metrics and success rates."""
    fe_data: dict[str, dict] = {
        fe: {
            "zlib_entropies": [],
            "bigram_entropies": [],
            "monogram_entropies": [],
            "chars": [],
            "passes": 0,
            "total": 0,
        }
        for fe in frontend_labels
    }

    for r in results:
        fe = r.get("frontend", "")
        enc = r.get("encoded", "")
        if fe not in fe_data or not enc:
            continue
        d = fe_data[fe]
        d["zlib_entropies"].append(zlib_entropy(enc))
        d["bigram_entropies"].append(ngram_entropy(enc, 2))
        d["monogram_entropies"].append(monogram_entropy(enc))
        d["chars"].append(len(enc))
        d["total"] += 1
        if r.get("first_pass"):
            d["passes"] += 1

    stats: dict[str, dict] = {}
    for fe in frontend_labels:
        d = fe_data[fe]
        if not d["zlib_entropies"]:
            continue
        success_rate = d["passes"] / d["total"] if d["total"] else 0.0
        stats[fe] = {
            "zlib_entropy": sum(d["zlib_entropies"]) / len(d["zlib_entropies"]),
            "bigram_entropy": sum(d["bigram_entropies"]) / len(d["bigram_entropies"]),
            "monogram_entropy": sum(d["monogram_entropies"]) / len(d["monogram_entropies"]),
            "avg_chars": sum(d["chars"]) / len(d["chars"]),
            "success_rate": success_rate,
            "n": d["total"],
        }
    return stats


def generate_entropy_analysis() -> None:
    # ── Load data ──────────────────────────────────────────────────
    phase2_results = _load_jsonl_dir(RAW_DIR, MODELS)
    if not phase2_results:
        print("Error: no Phase 2 results found", file=sys.stderr)
        sys.exit(1)

    # Noise robustness sweep (random char deletion — NOT compression)
    noise_results = _load_jsonl_dir(NOISE_SWEEP_DIR, SWEEP_MODELS)
    noise_labels = sorted(
        {r["frontend"] for r in noise_results if r.get("frontend", "").startswith("code_del")}
    ) if noise_results else []

    # Semantic compression sweep (stop-word removal — IS compression)
    comp_results = _load_jsonl_dir(COMP_SWEEP_DIR, SWEEP_MODELS)
    comp_labels = sorted(
        {r["frontend"] for r in comp_results if r.get("frontend", "").startswith("nl_stop")}
    ) if comp_results else []

    all_labels = FRONTENDS + noise_labels + comp_labels
    all_results = phase2_results + noise_results + comp_results

    print(
        f"  Phase 2: {len(phase2_results)} runs | "
        f"Noise sweep: {len(noise_results)} runs ({len(noise_labels)} levels) | "
        f"Compression sweep: {len(comp_results)} runs ({len(comp_labels)} levels)",
        file=sys.stderr,
    )

    stats = _compute_stats(all_results, all_labels)

    # ── Spearman correlation (Phase 2 frontends only) ─────────────
    p2_fe_present = [fe for fe in FRONTENDS if fe in stats]
    p2_zlib = [stats[fe]["zlib_entropy"] for fe in p2_fe_present]
    p2_succ = [stats[fe]["success_rate"] for fe in p2_fe_present]
    rho_zlib, p_zlib = _spearman_rho(p2_zlib, p2_succ)
    p2_mono = [stats[fe]["monogram_entropy"] for fe in p2_fe_present]
    rho_mono, p_mono = _spearman_rho(p2_mono, p2_succ)

    sig_zlib = "significant" if p_zlib < 0.05 else "NOT significant"
    sig_mono = "significant" if p_mono < 0.05 else "NOT significant"

    # ── Write CSV ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "entropy_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frontend",
            "zlib_entropy_bits_per_char",
            "bigram_entropy_bits_per_char",
            "monogram_entropy_bits_per_char",
            "avg_chars",
            "success_rate",
            "n",
            "category",  # phase2 | noise_robustness | compression
        ])
        for fe in all_labels:
            if fe not in stats:
                continue
            s = stats[fe]
            if fe in FRONTENDS:
                cat = "phase2"
            elif fe.startswith("code_del"):
                cat = "noise_robustness"
            elif fe.startswith("nl_stop"):
                cat = "compression"
            else:
                cat = "other"
            writer.writerow([
                fe,
                f"{s['zlib_entropy']:.4f}",
                f"{s['bigram_entropy']:.4f}",
                f"{s['monogram_entropy']:.4f}",
                f"{s['avg_chars']:.1f}",
                f"{s['success_rate']*100:.1f}%",
                s["n"],
                cat,
            ])
    print(f"  CSV: {csv_path}", file=sys.stderr)

    # ── Print summary ──────────────────────────────────────────────
    print(f"\n{'='*100}", file=sys.stderr)
    print(f"Entropy Analysis (v3): Compression-Based Entropy", file=sys.stderr)
    print(f"  Exploratory — no threshold claims, no relationship claims.", file=sys.stderr)
    print(f"{'='*100}", file=sys.stderr)

    # ── Section 1: Metric comparison ──────────────────────────────
    print(f"\n  [1] Metric Comparison: zlib DEFLATE vs monogram Shannon", file=sys.stderr)
    print(f"      (This is a methodological observation, NOT a finding about", file=sys.stderr)
    print(f"       entropy-success relationships.)", file=sys.stderr)
    print(f"\n  {'Frontend':<14} {'zlib H':>8} {'BiH':>8} {'MonoH':>8} {'Δ(Mono−zlib)':>14}", file=sys.stderr)
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*14}", file=sys.stderr)
    for fe in FRONTENDS:
        if fe not in stats:
            continue
        s = stats[fe]
        delta = s["monogram_entropy"] - s["zlib_entropy"]
        print(
            f"  {fe:<14} {s['zlib_entropy']:>8.3f} {s['bigram_entropy']:>8.3f}"
            f" {s['monogram_entropy']:>8.3f} {delta:>+14.3f}",
            file=sys.stderr,
        )
    print(
        f"\n  Observation: monogram entropy assigns nearly identical values\n"
        f"  to code (4.43) and json (4.44), masking json's higher redundancy.\n"
        f"  zlib correctly identifies json as lower-entropy (more redundant)\n"
        f"  due to its repetitive punctuation structure.",
        file=sys.stderr,
    )

    # ── Section 2: Correlation (explicitly NOT significant) ───────
    print(f"\n  [2] Correlation: DEFLATE entropy vs success rate (n=5)", file=sys.stderr)
    print(f"      Spearman ρ = {rho_zlib:+.2f}, p = {p_zlib:.2f} → {sig_zlib}", file=sys.stderr)
    print(f"      (monogram:  ρ = {rho_mono:+.2f}, p = {p_mono:.2f} → {sig_mono})", file=sys.stderr)
    print(
        f"\n  ⚠ 5 data points cannot establish a relationship. The correlation\n"
        f"  is reported for transparency only — it is noise-level and must\n"
        f"  NOT be cited as evidence of any entropy-success relationship.\n"
        f"  The 'H ≈ 4.2 warning line' from v1 is formally withdrawn.",
        file=sys.stderr,
    )

    # ── Section 3: Noise robustness (if available) ────────────────
    if noise_labels:
        print(f"\n  [3] Character-Level Noise Robustness (random char deletion)", file=sys.stderr)
        print(f"      ⚠ This is a DIFFERENT experiment from compression testing.", file=sys.stderr)
        print(f"      Random deletion tests form-corruption resilience, NOT", file=sys.stderr)
        print(f"      information-loss tolerance. Do NOT conflate with compression.", file=sys.stderr)
        print(f"\n  {'Frontend':<16} {'zlib H':>8} {'Chars':>7} {'Success':>8}", file=sys.stderr)
        print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8}", file=sys.stderr)
        for fe in noise_labels:
            if fe not in stats:
                continue
            s = stats[fe]
            print(
                f"  {fe:<16} {s['zlib_entropy']:>8.3f} {s['avg_chars']:>7.1f}"
                f" {s['success_rate']*100:>7.1f}%",
                file=sys.stderr,
            )
    else:
        print(f"\n  [3] Noise robustness sweep: not yet run (run_entropy_sweep.py)", file=sys.stderr)

    # ── Section 4: Semantic compression (if available) ─────────────
    if comp_labels:
        print(f"\n  [4] Semantic Compression Sweep (stop-word removal)", file=sys.stderr)
        print(f"      This IS a valid compression-strength variable, comparable", file=sys.stderr)
        print(f"      to llmlingua2's approach (removes low-information tokens first).", file=sys.stderr)
        print(f"\n  {'Frontend':<16} {'zlib H':>8} {'Chars':>7} {'Success':>8}", file=sys.stderr)
        print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8}", file=sys.stderr)
        comp_sorted = sorted(
            [(fe, stats[fe]) for fe in comp_labels if fe in stats],
            key=lambda x: x[1]["zlib_entropy"],
        )
        for fe, s in comp_sorted:
            print(
                f"  {fe:<16} {s['zlib_entropy']:>8.3f} {s['avg_chars']:>7.1f}"
                f" {s['success_rate']*100:>7.1f}%",
                file=sys.stderr,
            )

        # Spearman on compression sweep (more points = potentially meaningful)
        comp_zlib = [s["zlib_entropy"] for _, s in comp_sorted]
        comp_succ = [s["success_rate"] for _, s in comp_sorted]
        if len(comp_sorted) >= 5:
            rho_c, p_c = _spearman_rho(comp_zlib, comp_succ)
            sig_c = "significant" if p_c < 0.05 else "NOT significant"
            print(
                f"\n  Spearman ρ = {rho_c:+.2f}, p = {p_c:.3f} → {sig_c} (n={len(comp_sorted)})",
                file=sys.stderr,
            )
    else:
        print(f"\n  [4] Semantic compression sweep: not yet run (run_compression_sweep.py)", file=sys.stderr)

    print(f"\n{'='*100}", file=sys.stderr)

    # ── Generate figures ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # ── Figure 1: zlib entropy vs success (Phase 2 only, no line) ─
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))

        for fe in FRONTENDS:
            if fe not in stats:
                continue
            s = stats[fe]
            x = s["zlib_entropy"]
            y = s["success_rate"] * 100
            ax.scatter(x, y, s=150, zorder=5, color="#2ecc71", edgecolors="black")
            ax.annotate(
                fe, (x, y), textcoords="offset points",
                xytext=(10, 5), fontsize=11, fontweight="bold",
            )

        # Annotate with correlation stats
        ax.text(
            0.05, 0.05,
            f"Spearman ρ = {rho_zlib:+.2f}, p = {p_zlib:.2f}\n"
            f"n = {len(p2_fe_present)} — NOT significant\n"
            f"No relationship claim",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

        # Semantic compression sweep (separate colour, connected)
        if comp_labels:
            comp_pts = []
            for fe in comp_labels:
                if fe not in stats:
                    continue
                s = stats[fe]
                comp_pts.append((s["zlib_entropy"], s["success_rate"] * 100, fe))
            comp_pts.sort(key=lambda p: p[0])
            if len(comp_pts) >= 2:
                xs = [p[0] for p in comp_pts]
                ys = [p[1] for p in comp_pts]
                ax.plot(xs, ys, "s-", color="#3498db", markersize=7, linewidth=2,
                        label="Semantic compression sweep", zorder=4)
                for x, y, fe in comp_pts:
                    ax.annotate(
                        fe, (x, y), textcoords="offset points",
                        xytext=(8, -10), fontsize=7, color="#3498db",
                    )

        # Noise robustness sweep (separate colour, clearly different)
        if noise_labels:
            noise_pts = []
            for fe in noise_labels:
                if fe not in stats:
                    continue
                s = stats[fe]
                noise_pts.append((s["zlib_entropy"], s["success_rate"] * 100, fe))
            noise_pts.sort(key=lambda p: p[0])
            if len(noise_pts) >= 2:
                xs = [p[0] for p in noise_pts]
                ys = [p[1] for p in noise_pts]
                ax.plot(xs, ys, "D--", color="#e74c3c", markersize=7, linewidth=1.5,
                        label="Noise robustness (NOT compression)", zorder=3)
                for x, y, fe in noise_pts:
                    ax.annotate(
                        fe, (x, y), textcoords="offset points",
                        xytext=(8, -10), fontsize=7, color="#e74c3c",
                    )

        ax.set_xlabel("DEFLATE Entropy (bits/char)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(
            "Entropy vs Semantic Fidelity\n"
            "Exploratory — no threshold claim (ρ not significant)",
            fontsize=13,
        )
        if noise_labels or comp_labels:
            ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = FIGURES_DIR / "entropy_curve.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  Figure: {fig_path}", file=sys.stderr)

        # ── Figure 2: metric comparison bar chart ──────────────────
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

        fe_list = [fe for fe in FRONTENDS if fe in stats]
        x_pos = list(range(len(fe_list)))
        bar_width = 0.25

        zlib_vals = [stats[fe]["zlib_entropy"] for fe in fe_list]
        bigram_vals = [stats[fe]["bigram_entropy"] for fe in fe_list]
        mono_vals = [stats[fe]["monogram_entropy"] for fe in fe_list]

        ax2.bar([x - bar_width for x in x_pos], zlib_vals, bar_width,
                label="DEFLATE (compression-based)", color="#2ecc71")
        ax2.bar(x_pos, bigram_vals, bar_width,
                label="Bigram entropy", color="#3498db")
        ax2.bar([x + bar_width for x in x_pos], mono_vals, bar_width,
                label="Monogram (overestimates)", color="#e74c3c")

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(fe_list, fontsize=11)
        ax2.set_ylabel("Entropy (bits/char)", fontsize=12)
        ax2.set_title("Entropy Metric Comparison: Why Monogram Misleads", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        fig2.tight_layout()
        fig2_path = FIGURES_DIR / "entropy_metric_comparison.png"
        fig2.savefig(fig2_path, dpi=150)
        print(f"  Figure: {fig2_path}", file=sys.stderr)

    except ImportError:
        print("  [skip] matplotlib not available", file=sys.stderr)


if __name__ == "__main__":
    generate_entropy_analysis()
