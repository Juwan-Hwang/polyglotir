"""Full data integrity check for phase3_heartbeat — all models."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HB_DIR = ROOT / "data" / "raw" / "phase3_heartbeat"

all_results = []
for jsonl in sorted(HB_DIR.glob("*/results.jsonl")):
    model = jsonl.parent.name
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    results = [json.loads(l) for l in lines]
    all_results.extend(results)
    print(f"\n{'='*60}")
    print(f"Model: {model}  ({len(results)} lines)")

    errors = [r for r in results if "error" in r.get("judge", "").lower() or r.get("judge") == "unrecoverable_error"]
    empty = [r for r in results if not r.get("model_response", "")]
    passes = sum(1 for r in results if r["first_pass"])
    print(f"  Errors: {len(errors)}  |  Empty: {len(empty)}  |  Pass: {passes}/{len(results)} = {passes/len(results)*100:.1f}%")

    n_counts = Counter(r["n_value"] for r in results)
    print(f"  N distribution: {dict(sorted(n_counts.items()))}")
    turn_counts = Counter(r["turn"] for r in results)
    print(f"  Turn range: {min(turn_counts)}–{max(turn_counts)}, per-turn: {set(turn_counts.values())}")
    seed_counts = Counter(r["seed"] for r in results)
    print(f"  Seed distribution: {dict(sorted(seed_counts.items()))}")

    keys = [f'{r["n_value"]}|{r["seed"]}|{r["turn"]}' for r in results]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    print(f"  Duplicates: {len(dupes)}")

print(f"\n{'='*60}")
print(f"TOTAL: {len(all_results)} results across {len(HB_DIR.glob('*/results.jsonl'))} models")
total_errors = sum(1 for r in all_results if "error" in r.get("judge", "").lower())
total_empty = sum(1 for r in all_results if not r.get("model_response", ""))
total_pass = sum(1 for r in all_results if r["first_pass"])
print(f"  Total errors: {total_errors}")
print(f"  Total empty:  {total_empty}")
print(f"  Total pass:   {total_pass}/{len(all_results)} = {total_pass/len(all_results)*100:.1f}%")
