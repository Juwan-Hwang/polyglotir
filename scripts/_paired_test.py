"""Paired significance test: code vs json success rates (per-case paired)."""
import json, sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "phase2"
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6", "longcat-2.0", "minimax-m2.7"]

# Load all results: {(case_id, frontend, model) -> first_pass}
results = {}
for model in MODELS:
    slug = model.replace(".", "-").replace("/", "-")
    for line in (RAW / slug / "results.jsonl").read_text(encoding="utf-8").strip().split("\n"):
        r = json.loads(line)
        results[(r["case_id"], r["frontend"], r["model"])] = r["first_pass"]

# Paired comparison: for each (case, model) pair, compare code vs json
pairs_code = []
pairs_json = []
for (case_id, fe, model), passed in results.items():
    if fe == "code":
        key = (case_id, model)
        pairs_code.append((key, passed))
    elif fe == "json":
        key = (case_id, model)
        pairs_json.append((key, passed))

# Match pairs
code_dict = dict(pairs_code)
json_dict = dict(pairs_json)
common_keys = sorted(set(code_dict) & set(json_dict))

n = len(common_keys)
code_passes = sum(1 for k in common_keys if code_dict[k])
json_passes = sum(1 for k in common_keys if json_dict[k])

# McNemar's test (paired binary outcomes)
# b = code pass but json fail
# c = code fail but json pass
b = sum(1 for k in common_keys if code_dict[k] and not json_dict[k])
c = sum(1 for k in common_keys if not code_dict[k] and json_dict[k])

print(f"Paired comparison: code vs json")
print(f"  N pairs:     {n}")
print(f"  code passes: {code_passes} ({code_passes/n*100:.1f}%)")
print(f"  json passes: {json_passes} ({json_passes/n*100:.1f}%)")
print(f"  Both pass:   {sum(1 for k in common_keys if code_dict[k] and json_dict[k])}")
print(f"  Both fail:   {sum(1 for k in common_keys if not code_dict[k] and not json_dict[k])}")
print(f"  code only:   {b}")
print(f"  json only:   {c}")

# McNemar's chi-squared (with continuity correction)
if b + c > 0:
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # p-value from chi-squared with 1 df
    # Use survival function approximation
    p_value = math.exp(-chi2 / 2)  # rough approximation for chi2_1
    print(f"\n  McNemar's test:")
    print(f"    b (code only) = {b}, c (json only) = {c}")
    print(f"    chi2 = {chi2:.4f}")
    print(f"    p ~= {p_value:.4f} (approx)")
    if p_value < 0.05:
        print(f"    -> Significant difference (p < 0.05)")
    else:
        print(f"    -> NOT significant (p >= 0.05)")
else:
    print(f"\n  McNemar's test: b+c=0, no discordant pairs (identical performance)")

# Also do all pairwise comparisons
print(f"\n{'='*60}")
print(f"All pairwise McNemar's tests:")
print(f"{'='*60}")
frontends = ["code", "json", "natural", "nl_json", "llmlingua2"]
for i, fe1 in enumerate(frontends):
    for fe2 in frontends[i+1:]:
        d1 = {(case_id, model): results.get((case_id, fe1, model)) for (case_id, fe, model) in results if fe == fe1}
        d2 = {(case_id, model): results.get((case_id, fe2, model)) for (case_id, fe, model) in results if fe == fe2}
        keys = sorted(set(d1) & set(d2))
        b = sum(1 for k in keys if d1[k] and not d2[k])
        c = sum(1 for k in keys if not d1[k] and d2[k])
        if b + c == 0:
            print(f"  {fe1:12s} vs {fe2:12s}: identical (b+c=0)")
        else:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p = math.exp(-chi2 / 2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            r1 = sum(1 for k in keys if d1[k]) / len(keys) * 100
            r2 = sum(1 for k in keys if d2[k]) / len(keys) * 100
            print(f"  {fe1:12s} vs {fe2:12s}: {r1:5.1f}% vs {r2:5.1f}%  b={b} c={c}  p~={p:.4f} {sig}")
