#!/usr/bin/env python
"""Smoke test runner — Phase 0.5 execution.

Runs the full smoke test matrix:

    10 IR cases × N frontends × M models

For each (case, frontend, model) triple:
1. Compile the IR to the frontend's surface string.
2. Send to the model with a fixed decode prompt.
3. Judge the model's response (rule-based or LLM).
4. Record results to JSONL in ``data/raw/phase0.5/``.

After all runs, generates summary CSV in ``data/processed/phase0.5/``.

Per spec §4 Phase 0.5:
- Each IR is compiled ONCE (no per-model tuning).
- Record first-pass raw success rate.
- Judge uses LLM-as-judge (semantic, not string match).
- Results are written to compile.lock for audit.

Usage::

    # Rule-based judge, local models only (fast, no API needed)
    python scripts/run_smoke_test.py --models smollm-360m --judge rule

    # LLM judge, all configured models
    python scripts/run_smoke_test.py --judge llm

    # Specific frontends only
    python scripts/run_smoke_test.py --frontends code natural

    # Dry run (show what would run, but don't call models)
    python scripts/run_smoke_test.py --dry-run
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLES_DIR = ROOT / "examples"
RAW_DIR = ROOT / "data" / "raw" / "phase0.5"
PROCESSED_DIR = ROOT / "data" / "processed" / "phase0.5"

# ── Prompt template ───────────────────────────────────────────────────
# Fixed prompt — no per-model tuning. Per spec: "禁针对失败模型特供"
DECODE_PROMPT = """Decode the following SILP payload and explain what action(s) should be taken. Describe the full intent including all conditions, entities, and alternatives.

SILP payload:
{encoded}

Explain the semantic intent:"""


# ── Core runner ───────────────────────────────────────────────────────


def load_task_set() -> list[tuple[str, dict]]:
    """Load all example IR files, sorted by name."""
    from silp.ir import validate as validate_ir

    cases = []
    for path in sorted(EXAMPLES_DIR.glob("case*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result = validate_ir(data)
        if not result.valid:
            print(f"  [skip] {path.name}: {result.errors}", file=sys.stderr)
            continue
        cases.append((path.stem, result.ir))
    return cases


def run_smoke_test(
    models: list[str],
    frontends: list[str],
    judge_mode: str = "rule",
    judge_model: str = "gpt-4o-mini",
    dry_run: bool = False,
) -> None:
    """Run the full smoke test matrix.

    Args:
        models: List of model names to test.
        frontends: List of frontend names to test.
        judge_mode: "rule" or "llm".
        judge_model: LLM judge model name (if judge_mode="llm").
        dry_run: If True, show the matrix but don't call models.
    """
    from silp.bench.models import (
        GenerationConfig,
        get_model,
        list_models,
        load_env,
    )
    from silp.bench.judge import get_judge
    from silp.frontend import get_frontend as get_fe, list_frontends
    from silp.ir import validate as validate_ir

    load_env()

    # Validate model names
    available_models = list_models()
    for m in models:
        if m not in available_models:
            print(f"Error: unknown model {m!r}", file=sys.stderr)
            print(f"Available: {', '.join(available_models)}", file=sys.stderr)
            sys.exit(1)

    # Validate frontend names
    available_frontends = list_frontends()
    for f in frontends:
        if f not in available_frontends:
            print(f"Error: unknown frontend {f!r}", file=sys.stderr)
            print(f"Available: {', '.join(available_frontends)}", file=sys.stderr)
            sys.exit(1)

    # Load task set
    cases = load_task_set()
    if not cases:
        print("Error: no valid IR cases found in examples/", file=sys.stderr)
        sys.exit(1)

    total = len(cases) * len(frontends) * len(models)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SILP Smoke Test — Phase 0.5", file=sys.stderr)
    print(f"  Cases:     {len(cases)}", file=sys.stderr)
    print(f"  Frontends: {frontends}", file=sys.stderr)
    print(f"  Models:    {models}", file=sys.stderr)
    print(f"  Judge:     {judge_mode}" +
          (f" ({judge_model})" if judge_mode == "llm" else ""), file=sys.stderr)
    print(f"  Total:     {total} runs", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    if dry_run:
        print("Dry run — showing matrix without calling models:", file=sys.stderr)
        for case_id, ir in cases:
            for fe_name in frontends:
                fe = get_fe(fe_name)
                encoded = fe.compile(ir)
                for model_name in models:
                    print(f"  {case_id} | {fe_name} | {model_name}", file=sys.stderr)
                    print(f"    encoded: {encoded[:80]}...", file=sys.stderr)
        return

    # Prepare output directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize judge
    judge = get_judge(judge_mode, judge_model)

    # Pre-compile all IRs (each IR compiled ONCE per frontend)
    compiled: dict[tuple[str, str], str] = {}  # (case_id, frontend) → encoded
    for case_id, ir in cases:
        for fe_name in frontends:
            fe = get_fe(fe_name)
            compiled[(case_id, fe_name)] = fe.compile(ir)

    # Run the matrix
    all_results: list[dict[str, object]] = []
    pass_count = 0

    for model_name in models:
        # Open JSONL for this model
        model_slug = model_name.replace(".", "-").replace("/", "-")
        jsonl_path = RAW_DIR / model_slug / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        model = get_model(model_name)

        for case_id, ir in cases:
            for fe_name in frontends:
                encoded = compiled[(case_id, fe_name)]
                prompt = DECODE_PROMPT.format(encoded=encoded)

                run_id = f"{case_id}|{fe_name}|{model_name}"
                print(f"  [{len(all_results)+1}/{total}] {run_id}", end="",
                      file=sys.stderr, flush=True)

                # Generate
                response = model.generate(
                    prompt,
                    GenerationConfig(
                        max_new_tokens=256,
                        temperature=0.0,
                        timeout=30.0,
                    ),
                )

                if response.error:
                    print(f" → ERROR: {response.error[:60]}",
                          file=sys.stderr)
                    result = {
                        "case_id": case_id,
                        "frontend": fe_name,
                        "model": model_name,
                        "encoded": encoded,
                        "model_response": "",
                        "judge_verdict": "fail",
                        "judge_reason": f"Model error: {response.error}",
                        "judge": "error",
                        "elapsed": response.elapsed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": False,
                    }
                else:
                    # Judge
                    judge_result = judge.judge(ir, encoded, response.text)
                    if judge_result.passed:
                        pass_count += 1
                        print(f" → PASS ({response.elapsed:.1f}s)",
                              file=sys.stderr)
                    else:
                        print(f" → FAIL ({response.elapsed:.1f}s): "
                              f"{judge_result.reason[:60]}",
                              file=sys.stderr)

                    result = {
                        "case_id": case_id,
                        "frontend": fe_name,
                        "model": model_name,
                        "encoded": encoded,
                        "model_response": response.text,
                        "judge_verdict": judge_result.verdict,
                        "judge_reason": judge_result.reason,
                        "judge": judge_result.judge,
                        "elapsed": response.elapsed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": judge_result.passed,
                    }

                all_results.append(result)

                # Append to JSONL immediately (crash-safe)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Smoke test complete!", file=sys.stderr)
    print(f"  Total runs:  {len(all_results)}", file=sys.stderr)
    print(f"  Passed:      {pass_count}", file=sys.stderr)
    print(f"  Pass rate:   {pass_count/len(all_results)*100:.1f}%",
          file=sys.stderr)

    # Generate summary CSV
    _write_summary_csv(all_results)

    # Phase 0.5 gate check
    rate = pass_count / len(all_results) * 100 if all_results else 0
    print(f"\n  Phase 0.5 Gate:", file=sys.stderr)
    if rate >= 85:
        print(f"  ✓ {rate:.1f}% ≥ 85% → proceed to Phase 1", file=sys.stderr)
    elif rate >= 70:
        print(f"  ⚠ {rate:.1f}% (70–84%) → Phase 1, but prioritize "
              f"failure analysis", file=sys.stderr)
    else:
        print(f"  ✗ {rate:.1f}% < 70% → major IR/frontend revision needed",
              file=sys.stderr)

    print(f"\n  Raw JSONL:  {RAW_DIR}", file=sys.stderr)
    print(f"  Summary:    {PROCESSED_DIR}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def _write_summary_csv(results: list[dict[str, object]]) -> None:
    """Write per-(model, frontend) pass rate summary."""
    from collections import defaultdict

    # Aggregate: (model, frontend) → [pass/fail, ...]
    matrix: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        key = (r["model"], r["frontend"])
        matrix[key].append(r["first_pass"])

    # Write matrix CSV
    csv_path = PROCESSED_DIR / "success_rates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "frontend", "total", "passed",
                         "pass_rate", "cases_failed"])
        for (model, frontend), flags in sorted(matrix.items()):
            total = len(flags)
            passed = sum(flags)
            rate = passed / total * 100 if total else 0
            failed = [str(i+1) for i, ok in enumerate(flags) if not ok]
            writer.writerow([model, frontend, total, passed,
                            f"{rate:.1f}%", ";".join(failed)])

    # Write per-case detail
    detail_path = PROCESSED_DIR / "case_details.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "frontend", "model", "verdict",
                         "elapsed", "reason"])
        for r in results:
            writer.writerow([
                r["case_id"], r["frontend"], r["model"],
                r["judge_verdict"], f"{r['elapsed']:.2f}",
                r["judge_reason"][:100],
            ])


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SILP smoke test runner (Phase 0.5)"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model names to test (default: all local models). "
             "Available: smollm-360m, qwen2.5-0.5b, tinyllama-1.1b, "
             "gpt-4o-mini, claude-3.5-sonnet, gemini-pro",
    )
    parser.add_argument(
        "--frontends", nargs="*", default=None,
        help="Frontend names to test (default: all registered). "
             "Available: code, natural, json",
    )
    parser.add_argument(
        "--judge", choices=["rule", "llm"], default="rule",
        help="Judge mode: rule (fast, local) or llm (accurate, needs API)",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o-mini",
        help="Judge LLM model name (for --judge llm)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the test matrix without calling models",
    )
    args = parser.parse_args()

    # Defaults
    if args.models is None:
        # Default: local models only (safe, no API needed)
        args.models = ["smollm-360m", "qwen2.5-0.5b", "tinyllama-1.1b"]

    if args.frontends is None:
        from silp.frontend import list_frontends
        args.frontends = list_frontends()

    run_smoke_test(
        models=args.models,
        frontends=args.frontends,
        judge_mode=args.judge,
        judge_model=args.judge_model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
