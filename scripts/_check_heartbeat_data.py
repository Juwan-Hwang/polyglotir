"""Quick data integrity check for phase3_heartbeat results."""
import json
from collections import Counter

lines = open("data/raw/phase3_heartbeat/deepseek-v3-2/results.jsonl", encoding="utf-8").read().strip().split("\n")
results = [json.loads(l) for l in lines]

print(f"Total lines: {len(lines)}")

# 1. Error / empty check
errors = [r for r in results if "error" in r.get("judge", "") or r.get("judge") == "unrecoverable_error"]
empty = [r for r in results if not r.get("model_response", "")]
print(f"Error results: {len(errors)}")
print(f"Empty responses: {len(empty)}")

# 2. Pass rate
passes = sum(1 for r in results if r["first_pass"])
print(f"Pass: {passes}/{len(results)} = {passes / len(results) * 100:.1f}%")

# 3. N value distribution
n_counts = Counter(r["n_value"] for r in results)
print("\nN value distribution:")
for n in sorted(n_counts):
    p = sum(1 for r in results if r["n_value"] == n and r["first_pass"])
    t = n_counts[n]
    print(f"  N={n:>5}: {t:>3} runs, pass={p}/{t} = {p / t * 100:.1f}%")

# 4. Turn distribution
turn_counts = Counter(r["turn"] for r in results)
print(f"\nTurn distribution: turns 0–{max(turn_counts)}")
for t in sorted(turn_counts):
    print(f"  turn {t}: {turn_counts[t]} runs")

# 5. Seed distribution
seed_counts = Counter(r["seed"] for r in results)
print(f"\nSeed distribution: {dict(sorted(seed_counts.items()))}")

# 6. Context turns range
ctx = [r["context_turns"] for r in results]
print(f"\nContext turns range: {min(ctx)} – {max(ctx)}")

# 7. Duplicate keys
keys = [f'{r["n_value"]}|{r["seed"]}|{r["turn"]}' for r in results]
dupes = [k for k, c in Counter(keys).items() if c > 1]
print(f"Duplicate keys: {len(dupes)}")
if dupes:
    print(f"  Examples: {dupes[:10]}")

# 8. Check if all expected combos are present
# Expected: N values x seeds x turns
print(f"\nUnique N values: {sorted(set(r['n_value'] for r in results))}")
print(f"Unique seeds: {sorted(set(r['seed'] for r in results))}")
print(f"Unique turns: {sorted(set(r['turn'] for r in results))}")
expected = len(set(r["n_value"] for r in results)) * len(set(r["seed"] for r in results)) * len(set(r["turn"] for r in results))
print(f"Expected unique combos: {expected}")
print(f"Actual unique combos: {len(set(keys))}")
