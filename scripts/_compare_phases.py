#!/usr/bin/env python
"""Compare Phase 0.5 vs Phase 2 for the same model+frontend combos."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

models = ["deepseek-v3.2", "kimi-k2.6", "glm-5.2"]
frontends = ["code", "json", "natural"]

print(f"{'case_id':28s} {'frontend':10s} {'model':15s} {'P0.5':6s} {'P2':6s} {'delta'}")
print("-" * 80)

for m in models:
    slug = m.replace(".", "-")
    p05 = load(ROOT / f"data/raw/phase0.5/{slug}/results.jsonl")
    p2  = load(ROOT / f"data/raw/phase2/{slug}/results.jsonl")
    for fe in frontends:
        r05 = {r["case_id"]: r for r in p05 if r["frontend"] == fe}
        r2  = {r["case_id"]: r for r in p2  if r["frontend"] == fe}
        for case_id in sorted(r05.keys()):
            v05 = r05[case_id].get("judge_verdict", "?")
            v2  = r2.get(case_id, {}).get("judge_verdict", "?")
            delta = "FLIP!" if v05 != v2 else ""
            if delta:
                print(f"{case_id:28s} {fe:10s} {m:15s} {v05:6s} {v2:6s} {delta}")

# Summary
print("\n=== Summary ===")
for m in models:
    slug = m.replace(".", "-")
    p05 = load(ROOT / f"data/raw/phase0.5/{slug}/results.jsonl")
    p2  = load(ROOT / f"data/raw/phase2/{slug}/results.jsonl")
    for fe in frontends:
        r05 = [r for r in p05 if r["frontend"] == fe]
        r2  = [r for r in p2  if r["frontend"] == fe]
        rate05 = sum(1 for r in r05 if r.get("judge_verdict") == "pass") / len(r05) * 100 if r05 else 0
        rate2  = sum(1 for r in r2  if r.get("judge_verdict") == "pass") / len(r2)  * 100 if r2  else 0
        print(f"  {m:15s} {fe:10s}  P0.5={rate05:5.1f}%  P2={rate2:5.1f}%  delta={rate2-rate05:+.1f}%")
