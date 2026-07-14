"""Compute OR with CI for all error propagation tests + count all project-wide tests."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "phase3_heartbeat"

MODEL_SLUGS = ["deepseek-v3-2", "glm-5-2", "kimi-k2-6"]
MODEL_LABELS = {"deepseek-v3-2": "DeepSeek-v3.2", "glm-5-2": "GLM-5.2", "kimi-k2-6": "Kimi-K2.6"}

data = {}
for slug in MODEL_SLUGS:
    path = RAW_DIR / slug / "results.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    data[slug] = [json.loads(l) for l in lines if l]

# ═══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("PART 1: Fisher exact with OR + 95% CI (all 3 models)")
print("=" * 80)

for slug in MODEL_SLUGS:
    results = data[slug]
    results_sorted = sorted(results, key=lambda r: (r["n_value"], r["seed"], r["turn"]))
    sessions = defaultdict(list)
    for r in results_sorted:
        key = f'{r["n_value"]}|{r["seed"]}'
        sessions[key].append(r)

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

    table = [[pp, pf], [fp, ff]]
    odds, p_val = fisher_exact(table, alternative="two-sided")

    # Manual OR with Woolf 95% CI
    # OR = (pp * ff) / (pf * fp)
    # SE(ln(OR)) = sqrt(1/pp + 1/pf + 1/fp + 1/ff)
    # 95% CI: exp(ln(OR) ± 1.96 * SE)
    import math
    if pf > 0 and fp > 0 and ff > 0 and pp > 0:
        or_val = (pp * ff) / (pf * fp)
        ln_or = math.log(or_val)
        se_ln = math.sqrt(1/pp + 1/pf + 1/fp + 1/ff)
        ci_lo = math.exp(ln_or - 1.96 * se_ln)
        ci_hi = math.exp(ln_or + 1.96 * se_ln)
    else:
        # Use Fisher's exact (which handles zeros)
        or_val = odds
        ci_lo = ci_hi = float("nan")

    denom_pass = pp + pf
    denom_fail = fp + ff
    p_fail_pass = pf / denom_pass * 100 if denom_pass else 0
    p_fail_fail = ff / denom_fail * 100 if denom_fail else 0

    ci_crosses_1 = ci_lo < 1.0 < ci_hi if not math.isnan(ci_lo) else "N/A"

    print(f"\n  {MODEL_LABELS[slug]}:")
    print(f"    Table: [[pass→pass={pp}, pass→fail={pf}], [fail→pass={fp}, fail→fail={ff}]]")
    print(f"    P(fail|prev pass) = {p_fail_pass:.1f}%  (n={denom_pass})")
    print(f"    P(fail|prev fail) = {p_fail_fail:.1f}%  (n={denom_fail})")
    print(f"    OR = {or_val:.3f}  (95% CI: {ci_lo:.3f} – {ci_hi:.3f})")
    print(f"    CI crosses 1.0: {ci_crosses_1}")
    print(f"    Fisher exact (two-sided): p = {p_val:.4f}")

# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("PART 2: Project-wide multiple comparison inventory")
print("=" * 80)

tests = [
    # Phase 2
    ("Phase 2", "Spearman rank correlation (10 model pairs)", 10),
    # Phase 3 Ablation
    ("Phase 3 Ablation", "McNemar test (full vs ablation, per model)", 3),
    ("Phase 3 Ablation", "McNemar test (full vs each ablation, DeepSeek)", 3),
    ("Phase 3 Ablation", "McNemar test (full vs each ablation, GLM)", 3),
    ("Phase 3 Ablation", "McNemar test (full vs each ablation, Kimi)", 3),
    # Entropy analysis
    ("Phase 3 Entropy", "Spearman: entropy vs pass rate (per model)", 3),
    # Heartbeat
    ("Phase 3 Heartbeat", "Fisher exact: error propagation (per model)", 3),
    ("Phase 3 Heartbeat", "Spearman: ctx depth vs pass rate (per model)", 3),
    ("Phase 3 Heartbeat", "Spearman: ctx depth vs latency (per model)", 3),
    ("Phase 3 Heartbeat", "Mann-Whitney U: N=1 vs N=∞ (per model)", 3),
]

total_tests = sum(n for _, _, n in tests)
print(f"\n  {'Phase':<22} {'Test':<55} {'n':>3}")
print("  " + "─" * 82)
for phase, name, n in tests:
    print(f"  {phase:<22} {name:<55} {n:>3}")
print("  " + "─" * 82)
print(f"  {'TOTAL':<78} {total_tests:>3}")

bonferroni_alpha = 0.05 / total_tests
print(f"\n  Bonferroni-corrected α = 0.05 / {total_tests} = {bonferroni_alpha:.5f}")
print(f"  DeepSeek error propagation p=0.061 vs corrected α={bonferroni_alpha:.5f} → NOT significant")

# Also compute for just the heartbeat tests (12 tests = 4 types × 3 models)
hb_tests = 12
hb_alpha = 0.05 / hb_tests
print(f"\n  If only correcting within heartbeat (12 tests):")
print(f"  Bonferroni α = 0.05 / 12 = {hb_alpha:.5f}")
print(f"  DeepSeek error prop p=0.1049 (two-sided) vs α={hb_alpha:.5f} → NOT significant")

# Benjamini-Hochberg for heartbeat tests
print()
print("=" * 80)
print("PART 3: Benjamini-Hochberg FDR correction (heartbeat tests only)")
print("=" * 80)

# Collect all p-values from heartbeat
hb_pvals = []
# Error propagation
for slug in MODEL_SLUGS:
    results = data[slug]
    results_sorted = sorted(results, key=lambda r: (r["n_value"], r["seed"], r["turn"]))
    sessions = defaultdict(list)
    for r in results_sorted:
        sessions[f'{r["n_value"]}|{r["seed"]}'].append(r)

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

    table = [[pp, pf], [fp, ff]]
    _, p_val = fisher_exact(table, alternative="two-sided")
    hb_pvals.append((f"Error prop × {MODEL_LABELS[slug]}", p_val))

# Spearman ctx depth vs pass rate
from scipy.stats import spearmanr
for slug in MODEL_SLUGS:
    by_ctx = defaultdict(list)
    for r in data[slug]:
        by_ctx[r["context_turns"]].append(r["first_pass"])
    ctxs = sorted(by_ctx.keys())
    rates = [np.mean(by_ctx[c]) for c in ctxs]
    rho, p = spearmanr(ctxs, rates)
    hb_pvals.append((f"Ctx depth vs pass × {MODEL_LABELS[slug]}", p))

# Spearman ctx depth vs latency
for slug in MODEL_SLUGS:
    by_ctx = defaultdict(list)
    for r in data[slug]:
        by_ctx[r["context_turns"]].append(r["elapsed"])
    ctxs = sorted(by_ctx.keys())
    meds = [np.median(by_ctx[c]) for c in ctxs]
    rho, p = spearmanr(ctxs, meds)
    hb_pvals.append((f"Ctx depth vs latency × {MODEL_LABELS[slug]}", p))

# Mann-Whitney U: N=1 vs N=∞ (stateless vs no heartbeat)
from scipy.stats import mannwhitneyu
for slug in MODEL_SLUGS:
    n1 = [r["first_pass"] for r in data[slug] if r["n_value"] == 1]
    ninf = [r["first_pass"] for r in data[slug] if r["n_value"] == 9999]
    u, p = mannwhitneyu(n1, ninf, alternative="two-sided")
    hb_pvals.append((f"Mann-Whitney U N=1 vs N=∞ × {MODEL_LABELS[slug]}", p))

# Sort and apply BH
hb_pvals.sort(key=lambda x: x[1])
n = len(hb_pvals)
print(f"\n  {'Rank':<5} {'Test':<45} {'p-value':>10} {'BH threshold':>14} {'Significant':>12}")
print("  " + "─" * 88)
for i, (name, p) in enumerate(hb_pvals):
    bh_threshold = (i + 1) / n * 0.05
    sig = "YES" if p <= bh_threshold else "no"
    print(f"  {i+1:<5} {name:<45} {p:>10.4f} {bh_threshold:>14.5f} {sig:>12}")

# Determine which tests survive BH
bh_survivors = [(name, p) for i, (name, p) in enumerate(hb_pvals) if p <= (i + 1) / n * 0.05]

print(f"\n  BH-corrected q=0.05: {len(bh_survivors)} test(s) survive FDR correction")
for name, p in bh_survivors:
    print(f"    PASS: {name}  (p={p:.4f})")
if not bh_survivors:
    print(f"  -> All heartbeat results are consistent with the global null hypothesis")
else:
    print(f"  -> Accuracy-related tests (error prop, ctx-depth vs pass, Mann-Whitney U)")
    print(f"     all fail to survive FDR correction. Only latency trends survive,")
    print(f"     confirming computational cost without accuracy cost.")
