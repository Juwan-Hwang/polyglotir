"""Find missing (N, seed, turn) combos in kimi heartbeat data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lines = (ROOT / "data/raw/phase3_heartbeat/kimi-k2-6/results.jsonl").read_text("utf-8").strip().split("\n")
results = [json.loads(l) for l in lines if l]

existing = set()
for r in results:
    existing.add(f'{r["n_value"]}|{r["seed"]}|{r["turn"]}')

missing = []
for s in range(8):
    for n in [1, 5, 10, 15, 20, 9999]:
        for t in range(15):
            key = f"{n}|{s}|{t}"
            if key not in existing:
                missing.append((n, s, t))

print(f"Missing: {len(missing)}")
for m in missing:
    print(f"  N={m[0]} seed={m[1]} turn={m[2]}")
