"""Deep analysis of heartbeat data — context depth breakdown + trend tests."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "phase3_heartbeat"

MODEL_SLUGS = ["deepseek-v3-2", "glm-5-2", "kimi-k2-6"]
MODEL_LABELS = {"deepseek-v3-2": "DeepSeek-v3.2", "glm-5-2": "GLM-5.2", "kimi-k2-6": "Kimi-K2.6"}
N_VALUES = [1, 5, 10, 15, 20, 9999]

data = {}
for slug in MODEL_SLUGS:
    path = RAW_DIR / slug / "results.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    data[slug] = [json.loads(l) for l in lines if l]

print("=" * 70)
print("1. Pass rate by context depth (all N values pooled)")
print("=" * 70)
for slug in MODEL_SLUGS:
    results = data[slug]
    by_ctx = defaultdict(list)
    for r in results:
        by_ctx[r["context_turns"]].append(r["first_pass"])
    print(f"\n  {MODEL_LABELS[slug]}:")
    for ctx in sorted(by_ctx):
        vals = by_ctx[ctx]
        rate = np.mean(vals) * 100
        print(f"    ctx={ctx:>2}: {rate:5.1f}%  (n={len(vals)})")
    # Spearman: does pass rate correlate with context depth?
    ctxs = sorted(by_ctx.keys())
    rates = [np.mean(by_ctx[c]) for c in ctxs]
    rho, p = spearmanr(ctxs, rates)
    print(f"    Spearman ρ = {rho:.3f}, p = {p:.4f}")

print("\n" + "=" * 70)
print("2. Per-N pass rate by context depth")
print("=" * 70)
for slug in MODEL_SLUGS:
    results = data[slug]
    print(f"\n  {MODEL_LABELS[slug]}:")
    for n in N_VALUES:
        subset = [r for r in results if r["n_value"] == n]
        by_ctx = defaultdict(list)
        for r in subset:
            by_ctx[r["context_turns"]].append(r["first_pass"])
        ctxs = sorted(by_ctx.keys())
        rates_str = "  ".join(f"ctx{c}={np.mean(by_ctx[c])*100:.0f}%" for c in ctxs)
        n_label = "∞" if n == 9999 else str(n)
        print(f"    N={n_label:>4}: {rates_str}")

print("\n" + "=" * 70)
print("3. N=1 (stateless) vs N=∞ (no heartbeat) — Mann-Whitney U")
print("=" * 70)
for slug in MODEL_SLUGS:
    results = data[slug]
    n1 = [r["first_pass"] for r in results if r["n_value"] == 1]
    ninf = [r["first_pass"] for r in results if r["n_value"] == 9999]
    u, p = mannwhitneyu(n1, ninf, alternative="two-sided")
    print(f"  {MODEL_LABELS[slug]}: N=1 mean={np.mean(n1):.3f}, N=∞ mean={np.mean(ninf):.3f}, "
          f"U={u:.0f}, p={p:.4f}")

print("\n" + "=" * 70)
print("4. N=15/20/9999 are identical (heartbeat never fires in 15 turns)")
print("=" * 70)
for slug in MODEL_SLUGS:
    results = data[slug]
    for n in [15, 20, 9999]:
        subset = [r for r in results if r["n_value"] == n]
        ctxs = set(r["context_turns"] for r in subset)
        rate = np.mean([r["first_pass"] for r in subset]) * 100
        print(f"  {MODEL_LABELS[slug]} N={n}: pass={rate:.1f}%, context_turns={sorted(ctxs)}")

print("\n" + "=" * 70)
print("5. Latency by context depth (does accumulated context slow responses?)")
print("=" * 70)
for slug in MODEL_SLUGS:
    results = data[slug]
    by_ctx = defaultdict(list)
    for r in results:
        by_ctx[r["context_turns"]].append(r["elapsed"])
    print(f"\n  {MODEL_LABELS[slug]}:")
    for ctx in sorted(by_ctx):
        vals = by_ctx[ctx]
        print(f"    ctx={ctx:>2}: mean={np.mean(vals):5.1f}s  median={np.median(vals):5.1f}s  (n={len(vals)})")
