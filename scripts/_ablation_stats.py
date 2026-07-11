import json, collections, sys
from pathlib import Path

model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v3-2"
path = Path(f"data/raw/phase3_ablation/{model}/results.jsonl")
lines = path.read_text(encoding="utf-8").strip().split("\n")
d = collections.defaultdict(lambda: [0, 0])
for l in lines:
    r = json.loads(l)
    cond = r["condition"]
    d[cond][1] += 1
    if r.get("first_pass"):
        d[cond][0] += 1
print(f"=== {model} ({sum(v[1] for v in d.values())} runs so far) ===")
for k in sorted(d):
    v = d[k]
    print(f"  {k:12s} {v[0]}/{v[1]} ({v[0]/v[1]*100:.1f}%)")
