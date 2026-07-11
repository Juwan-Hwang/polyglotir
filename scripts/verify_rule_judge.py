"""Verify optimized RuleJudge against Phase 0.5 data.

Re-runs the RuleJudge on all Phase 0.5 results and compares:
1. Old rule verdict (from results.jsonl) vs new rule verdict
2. New rule verdict vs LLM verdict (agreement rate)
3. Lists fixed cases (old != llm, new == llm) and broken cases (old == llm, new != llm)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from silp.bench.judge import RuleJudge
from silp.ir.validator import validate as validate_ir

# ── Load case IRs from examples/ ─────────────────────────────────────

EXAMPLES_DIR = Path("examples")
ir_lookup: dict[str, dict] = {}

for path in sorted(EXAMPLES_DIR.glob("case*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = validate_ir(data)
    if result.valid:
        ir_lookup[path.stem] = data

print(f"Loaded {len(ir_lookup)} case IRs from examples/", file=sys.stderr)

# ── Run ──────────────────────────────────────────────────────────────

models_dirs = [
    "data/raw/phase0.5/deepseek-v3-2",
    "data/raw/phase0.5/kimi-k2-6",
    "data/raw/phase0.5/glm-5-2",
]

judge = RuleJudge()

total = 0
old_agree = 0
new_agree = 0
fixed: list[dict] = []
broken: list[dict] = []
still_wrong: list[dict] = []

for model_dir in models_dirs:
    results_file = os.path.join(model_dir, "results.jsonl")
    if not os.path.exists(results_file):
        continue
    model_name = os.path.basename(model_dir)

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if not rec.get("model_response"):
                continue
            if rec.get("llm_verdict") == "error":
                continue

            case_id = rec["case_id"]
            old_rule = rec.get("rule_verdict", "")
            llm = rec.get("llm_verdict", "")

            ir_dict = ir_lookup.get(case_id)
            if not ir_dict:
                continue

            result_val = validate_ir(ir_dict)
            if not result_val.valid:
                continue

            ir = result_val.ir
            result = judge.judge(ir, rec["encoded"], rec["model_response"])
            new_rule = result.verdict

            total += 1

            old_match = old_rule == llm
            new_match = new_rule == llm

            if old_match:
                old_agree += 1
            if new_match:
                new_agree += 1

            label = f"{case_id}|{rec['frontend']}|{model_name}"

            if not old_match and new_match:
                fixed.append({
                    "case": label,
                    "old": old_rule,
                    "new": new_rule,
                    "llm": llm,
                    "reason": result.reason,
                })
            elif old_match and not new_match:
                broken.append({
                    "case": label,
                    "old": old_rule,
                    "new": new_rule,
                    "llm": llm,
                    "reason": result.reason,
                })
            elif not old_match and not new_match:
                still_wrong.append({
                    "case": label,
                    "old": old_rule,
                    "new": new_rule,
                    "llm": llm,
                    "reason": result.reason,
                })

# ── Report ───────────────────────────────────────────────────────────

print("=== RuleJudge Optimization Verification ===")
print(f"Total cases (excl. errors): {total}")
print()
print(f"Old Rule Judge agreement with LLM: {old_agree}/{total} ({100*old_agree/total:.1f}%)")
print(f"New Rule Judge agreement with LLM: {new_agree}/{total} ({100*new_agree/total:.1f}%)")
print(f"Improvement: +{new_agree - old_agree} cases ({100*(new_agree - old_agree)/total:.1f}%)")
print()

if fixed:
    print(f"=== Fixed Cases ({len(fixed)}) ===")
    for fc in fixed:
        print(f"  {fc['case']}: {fc['old']} -> {fc['new']} (llm={fc['llm']})")
        print(f"    reason: {fc['reason']}")

if broken:
    print(f"\n=== Broken Cases ({len(broken)}) ===")
    for bc in broken:
        print(f"  {bc['case']}: {bc['old']} -> {bc['new']} (llm={bc['llm']})")
        print(f"    reason: {bc['reason']}")
else:
    print("\nNo regression (no previously-correct cases broken).")

if still_wrong:
    print(f"\n=== Still Wrong ({len(still_wrong)}) -- inherent rule limitations ===")
    for sw in still_wrong:
        print(f"  {sw['case']}: {sw['old']} -> {sw['new']} (llm={sw['llm']})")
        print(f"    reason: {sw['reason']}")
