import json, collections, math, sys
from pathlib import Path

RAW = Path("data/raw/phase3_ablation")
MODELS = ["deepseek-v3-2", "glm-5-2", "kimi-k2-6"]
CONDITIONS = ["original", "shuffled", "skeleton"]

# Order matters: longer prefixes first (case10 before case1)
CASE_TYPES = [
    ("case10", "multi_turn"),
    ("case1", "multi_constraint"),
    ("case2", "negation"),
    ("case3", "detail"),
    ("case5", "tool_branch"),
    ("case6", "nested_constraint"),
    ("case7", "parallel_action"),
    ("case8", "conditional_branch"),
    ("case9", "tool_call"),
]

def get_case_type(case_id):
    for prefix, ctype in CASE_TYPES:
        if case_id.startswith(prefix):
            return ctype
    return "unknown"

all_results = []
for slug in MODELS:
    for line in (RAW / slug / "results.jsonl").read_text("utf-8").strip().split("\n"):
        all_results.append(json.loads(line))

# Overall
print("=" * 70)
print("OVERALL (243 runs, 3 models)")
print("=" * 70)
by_cond = collections.defaultdict(lambda: [0, 0])
for r in all_results:
    by_cond[r["condition"]][1] += 1
    if r["first_pass"]: by_cond[r["condition"]][0] += 1
for c in CONDITIONS:
    p, t = by_cond[c]
    print(f"  {c:12s}  {p}/{t}  ({p/t*100:.1f}%)")

# Per case type
print()
print("=" * 70)
print("PER CASE TYPE (all models pooled, 9 runs per cell)")
print("=" * 70)
by_tc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in all_results:
    ct = get_case_type(r["case_id"])
    by_tc[ct][r["condition"]][1] += 1
    if r["first_pass"]: by_tc[ct][r["condition"]][0] += 1

print(f"  {'Type':<22} {'original':>10} {'shuffled':>10} {'skeleton':>10}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
for ct in sorted(by_tc):
    row = []
    for c in CONDITIONS:
        p, t = by_tc[ct][c]
        row.append(f"{p}/{t} ({p/t*100:.0f}%)" if t else "N/A")
    print(f"  {ct:<22} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# Negation per model
print()
print("=" * 70)
print("NEGATION CASES (core hypothesis) per model")
print("=" * 70)
neg = [r for r in all_results if get_case_type(r["case_id"]) == "negation"]
print(f"  {'Model':<16} {'original':>10} {'shuffled':>10} {'skeleton':>10}")
print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
MODEL_KEYS = {"deepseek-v3-2": "deepseek-v3.2", "glm-5-2": "glm-5.2", "kimi-k2-6": "kimi-k2.6"}
for slug in MODELS:
    mk = MODEL_KEYS.get(slug, slug)
    row = []
    for c in CONDITIONS:
        s = [r for r in neg if r["model"] == mk and r["condition"] == c]
        p = sum(1 for r in s if r["first_pass"])
        t = len(s)
        row.append(f"{p}/{t} ({p/t*100:.0f}%)" if t else "N/A")
    print(f"  {slug:<16} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# McNemar
print()
print("=" * 70)
print("MCNEMAR'S TESTS (paired by case x model)")
print("=" * 70)
lookup = {}
for r in all_results:
    lookup[((r["case_id"], r["model"]), r["condition"])] = r["first_pass"]
keys = sorted({(r["case_id"], r["model"]) for r in all_results})
for a, b in [("original", "shuffled"), ("original", "skeleton"), ("shuffled", "skeleton")]:
    bb = cc = 0
    for k in keys:
        pa = lookup.get((k, a))
        pb = lookup.get((k, b))
        if pa is None or pb is None: continue
        if pa and not pb: bb += 1
        elif not pa and pb: cc += 1
    n = bb + cc
    if n == 0:
        print(f"  {a:12s} vs {b:12s}: identical")
    else:
        chi2 = (abs(bb - cc) - 1) ** 2 / n
        p = math.erfc(math.sqrt(chi2 / 2.0))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {a:12s} vs {b:12s}: b={bb} c={cc} chi2={chi2:.4f} p={p:.6f} {sig}")
