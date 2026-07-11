#!/usr/bin/env python
"""Phase 2 benchmark matrix — frontend x model relative success-rate matrix.

Per spec section 4 Phase 2: "frontend x model relative success-rate matrix"

This is the core experiment of the paper. It runs the full matrix:

    N IR cases x F frontends x M models

For each triple:
1. Compile the IR with the frontend (each IR compiled ONCE per frontend).
2. Send to the model with a fixed decode prompt.
3. Judge the response (LLM-as-judge, semantic, not string match).
4. Record: verdict, latency, retries, raw response.

After all runs, generates:
- phase2_matrix.csv          -- F x M success-rate matrix (paper Table 1)
- phase2_matrix_transposed.csv -- M x F view (model-centric)
- phase2_case_details.csv    -- per-case detail for failure analysis
- phase2_retries.csv         -- retry audit log
- phase2_spearman.json       -- Spearman rank correlation between model pairs

Usage::

    # Full matrix: all frontends x all proxy models, LLM judge
    python scripts/run_phase2_matrix.py --judge llm

    # Subset for quick testing
    python scripts/run_phase2_matrix.py --frontends code json natural --models deepseek-v3.2 glm-5.2 --judge rule

    # Dry run
    python scripts/run_phase2_matrix.py --dry-run
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLES_DIR = ROOT / "examples"
RAW_DIR = ROOT / "data" / "raw" / "phase2"
PROCESSED_DIR = ROOT / "data" / "processed" / "phase2"

DECODE_PROMPT = """Decode the following SILP payload and explain what action(s) should be taken. Describe the full intent including all conditions, entities, and alternatives.

SILP payload:
{encoded}

Explain the semantic intent:"""


def load_task_set() -> list[tuple[str, object]]:
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


def run_matrix(
    models: list[str],
    frontends: list[str],
    judge_mode: str = "dual",
    judge_model: str = "glm-5.2",
    dry_run: bool = False,
) -> None:
    from silp.bench.models import GenerationConfig, get_model, list_models, load_env
    from silp.bench.judge import get_judge
    from silp.frontend import CompileLock, get_frontend as get_fe, list_frontends

    load_env()

    # Validate names
    available_models = list_models()
    for m in models:
        if m not in available_models:
            print(f"Error: unknown model {m!r}", file=sys.stderr)
            sys.exit(1)

    available_frontends = list_frontends()
    for f in frontends:
        if f not in available_frontends:
            print(f"Error: unknown frontend {f!r}", file=sys.stderr)
            sys.exit(1)

    cases = load_task_set()
    if not cases:
        print("Error: no valid IR cases found", file=sys.stderr)
        sys.exit(1)

    total = len(cases) * len(frontends) * len(models)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"SILP Phase 2 - Universality Benchmark Matrix", file=sys.stderr)
    print(f"  Cases:     {len(cases)}", file=sys.stderr)
    print(f"  Frontends: {frontends}", file=sys.stderr)
    print(f"  Models:    {models}", file=sys.stderr)
    print(f"  Judge:     {judge_mode}" + (f" ({judge_model})" if judge_mode in ("llm", "dual") else ""), file=sys.stderr)
    print(f"  Total:     {total} runs", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    if dry_run:
        print("Dry run:", file=sys.stderr)
        for case_id, ir in cases:
            for fe_name in frontends:
                fe = get_fe(fe_name)
                try:
                    encoded = fe.compile(ir)
                    print(f"  {case_id} | {fe_name} -> {encoded[:80]}...", file=sys.stderr)
                except Exception as e:
                    print(f"  {case_id} | {fe_name} -> COMPILE ERROR: {e}", file=sys.stderr)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    judge = get_judge(judge_mode, judge_model)

    # Pre-compile all IRs
    compiled: dict[tuple[str, str], str] = {}
    lock_path = RAW_DIR / "compile.lock.jsonl"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        for case_id, ir in cases:
            for fe_name in frontends:
                fe = get_fe(fe_name)
                try:
                    encoded = fe.compile(ir)
                except Exception as exc:
                    print(f"  [error] compile {case_id}/{fe_name}: {exc}", file=sys.stderr)
                    encoded = ""
                compiled[(case_id, fe_name)] = encoded
                if encoded:
                    lock = CompileLock.seal(frontend_name=fe_name, ir=ir, compiled=encoded)
                    lock_file.write(lock.to_json() + "\n")
    print(f"  Compile locks: {lock_path}", file=sys.stderr)

    # Run the matrix
    all_results: list[dict[str, object]] = []
    pass_count = 0
    retry_total = 0

    for model_name in models:
        model_slug = model_name.replace(".", "-").replace("/", "-")
        jsonl_path = RAW_DIR / model_slug / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

        model = get_model(model_name)

        for case_id, ir in cases:
            for fe_name in frontends:
                encoded = compiled.get((case_id, fe_name), "")
                if not encoded:
                    result = {
                        "case_id": case_id, "frontend": fe_name, "model": model_name,
                        "encoded": "", "model_response": "",
                        "judge_verdict": "fail",
                        "judge_reason": "Compile error",
                        "judge": "compile_error",
                        "elapsed": 0.0, "retries": 0,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": False,
                    }
                    all_results.append(result)
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                prompt = DECODE_PROMPT.format(encoded=encoded)
                run_id = f"{case_id}|{fe_name}|{model_name}"
                print(f"  [{len(all_results)+1}/{total}] {run_id}", end="", file=sys.stderr, flush=True)

                # ── Outer retry loop: keep retrying on infra errors ──
                # The model backend already retries internally (max_retries=5),
                # but if it STILL fails, we retry the whole call again.
                # This ensures no error results are ever recorded — the run
                # only proceeds once the model produces a real response.
                outer_retries = 0
                response = None
                while True:
                    response = model.generate(
                        prompt,
                        GenerationConfig(max_new_tokens=256, temperature=0.0, timeout=30.0),
                    )
                    if response.retries:
                        retry_total += response.retries

                    if response.error:
                        outer_retries += 1
                        print(
                            f" -> ERROR (outer retry #{outer_retries}): "
                            f"{response.error[:60]}",
                            file=sys.stderr,
                        )
                        if outer_retries < 10:
                            backoff = min(30, 5 * outer_retries)
                            time.sleep(backoff)
                            print(f"  retrying {run_id}...", end="", file=sys.stderr, flush=True)
                            continue
                        # Last resort: after 10 outer retries, force one more
                        # with a longer timeout
                        print(f"  final attempt with extended timeout...", end="", file=sys.stderr, flush=True)
                        response = model.generate(
                            prompt,
                            GenerationConfig(max_new_tokens=256, temperature=0.0, timeout=60.0),
                        )
                        if response.retries:
                            retry_total += response.retries
                        if response.error:
                            # This should essentially never happen. If it does,
                            # record it as a last resort but flag it loudly.
                            print(f" -> UNRECOVERABLE ERROR after {outer_retries} retries", file=sys.stderr)
                        break
                    break

                if response.error:
                    result = {
                        "case_id": case_id, "frontend": fe_name, "model": model_name,
                        "encoded": encoded, "model_response": "",
                        "judge_verdict": "fail",
                        "judge_reason": f"Model error (after {outer_retries} outer retries): {response.error}",
                        "judge": "unrecoverable_error",
                        "elapsed": response.elapsed, "retries": response.retries + outer_retries,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": False,
                    }
                else:
                    judge_result = judge.judge(ir, encoded, response.text)
                    if judge_result.passed:
                        pass_count += 1
                        print(f" -> PASS ({response.elapsed:.1f}s)", file=sys.stderr)
                    else:
                        print(f" -> FAIL ({response.elapsed:.1f}s): {judge_result.reason[:60]}", file=sys.stderr)

                    details = judge_result.details
                    rule_verdict = details.get("rule_verdict") if details else None
                    rule_reason = details.get("rule_reason") if details else None
                    llm_verdict = details.get("llm_verdict") if details else None
                    llm_reason = details.get("llm_reason") if details else None

                    result = {
                        "case_id": case_id, "frontend": fe_name, "model": model_name,
                        "encoded": encoded, "model_response": response.text,
                        "judge_verdict": judge_result.verdict,
                        "judge_reason": judge_result.reason,
                        "judge": judge_result.judge,
                        "rule_verdict": rule_verdict, "rule_reason": rule_reason,
                        "llm_verdict": llm_verdict, "llm_reason": llm_reason,
                        "elapsed": response.elapsed, "retries": response.retries + outer_retries,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": judge_result.passed,
                    }

                all_results.append(result)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    rate = pass_count / len(all_results) * 100 if all_results else 0
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Phase 2 matrix complete!", file=sys.stderr)
    print(f"  Total runs:    {len(all_results)}", file=sys.stderr)
    print(f"  Passed:        {pass_count}", file=sys.stderr)
    print(f"  Pass rate:     {rate:.1f}%", file=sys.stderr)
    print(f"  Retries used:  {retry_total}", file=sys.stderr)

    _write_matrix_csv(all_results, frontends, models)
    _write_case_details(all_results)
    _write_retry_audit(all_results)
    spearman = _compute_spearman(all_results, frontends, models)
    _write_spearman(spearman)

    n_points = len(cases) * len(frontends)
    print(f"\n  Data points per model pair: {n_points}", file=sys.stderr)
    if n_points >= 30:
        print(f"  {n_points} points - Spearman correlation is valid (threshold=30)", file=sys.stderr)
    else:
        print(f"  WARNING: {n_points} < 30 - need more cases/frontends", file=sys.stderr)

    print(f"\n  Raw JSONL:       {RAW_DIR}", file=sys.stderr)
    print(f"  Matrix CSV:      {PROCESSED_DIR}/phase2_matrix.csv", file=sys.stderr)
    print(f"  Case details:    {PROCESSED_DIR}/phase2_case_details.csv", file=sys.stderr)
    print(f"  Retry audit:     {PROCESSED_DIR}/phase2_retries.csv", file=sys.stderr)
    print(f"  Spearman:        {PROCESSED_DIR}/phase2_spearman.json", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)


def _write_matrix_csv(results, frontends, models) -> None:
    matrix: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        matrix[(r["frontend"], r["model"])].append(r["first_pass"])

    csv_path = PROCESSED_DIR / "phase2_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frontend"] + models + ["avg", "rank"])

        fe_avgs: dict[str, float] = {}
        for fe in frontends:
            all_flags: list[bool] = []
            for m in models:
                all_flags.extend(matrix.get((fe, m), []))
            fe_avgs[fe] = sum(all_flags) / len(all_flags) * 100 if all_flags else 0

        ranked = sorted(frontends, key=lambda fe: fe_avgs[fe], reverse=True)
        for fe in ranked:
            row = [fe]
            for m in models:
                flags = matrix.get((fe, m), [])
                row.append(f"{sum(flags)/len(flags)*100:.1f}%" if flags else "N/A")
            row.append(f"{fe_avgs[fe]:.1f}%")
            rank = sum(1 for other in frontends if fe_avgs[other] > fe_avgs[fe]) + 1
            row.append(rank)
            writer.writerow(row)

    # Transposed view
    csv_path_t = PROCESSED_DIR / "phase2_matrix_transposed.csv"
    with open(csv_path_t, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + frontends + ["avg"])
        for m in models:
            row = [m]
            all_flags: list[bool] = []
            for fe in frontends:
                flags = matrix.get((fe, m), [])
                all_flags.extend(flags)
                row.append(f"{sum(flags)/len(flags)*100:.1f}%" if flags else "N/A")
            avg = sum(all_flags) / len(all_flags) * 100 if all_flags else 0
            row.append(f"{avg:.1f}%")
            writer.writerow(row)


def _write_case_details(results) -> None:
    csv_path = PROCESSED_DIR / "phase2_case_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "frontend", "model", "llm_verdict", "rule_verdict", "elapsed", "retries", "llm_reason", "rule_reason"])
        for r in results:
            writer.writerow([
                r["case_id"], r["frontend"], r["model"],
                r.get("llm_verdict") or r["judge_verdict"],
                r.get("rule_verdict") or "",
                f"{r['elapsed']:.2f}", r.get("retries", 0),
                (r.get("llm_reason") or r["judge_reason"])[:120],
                (r.get("rule_reason") or "")[:120],
            ])


def _write_retry_audit(results) -> None:
    csv_path = PROCESSED_DIR / "phase2_retries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "frontend", "model", "retries", "verdict", "elapsed", "reason"])
        for r in results:
            if r.get("retries", 0) > 0:
                writer.writerow([
                    r["case_id"], r["frontend"], r["model"],
                    r["retries"], r["judge_verdict"],
                    f"{r['elapsed']:.2f}", r.get("judge_reason", "")[:120],
                ])


def _compute_spearman(results, frontends, models) -> dict:
    """Compute Spearman rank correlation between model pairs.

    Uses per-case, per-frontend binary outcomes (pass=1, fail=0) as
    paired data points -- not aggregated frontend rates.  This gives
    n_cases x n_frontends data points per model pair.
    """
    model_outcomes: dict[str, dict[tuple[str, str], int]] = {}
    for m in models:
        model_outcomes[m] = {}
        for r in results:
            if r["model"] == m:
                key = (r["case_id"], r["frontend"])
                model_outcomes[m][key] = 1 if r["first_pass"] else 0

    model_fe_rates: dict[str, dict[str, float]] = {}
    for m in models:
        model_fe_rates[m] = {}
        for fe in frontends:
            flags = [r["first_pass"] for r in results if r["model"] == m and r["frontend"] == fe]
            model_fe_rates[m][fe] = sum(flags) / len(flags) if flags else 0.0

    pairs: list[dict] = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            common_keys = sorted(
                set(model_outcomes[m1].keys()) & set(model_outcomes[m2].keys())
            )
            v1 = [model_outcomes[m1][k] for k in common_keys]
            v2 = [model_outcomes[m2][k] for k in common_keys]
            rho = _spearman_rho(v1, v2)
            pairs.append({
                "model_a": m1, "model_b": m2,
                "spearman_rho": round(rho, 4),
                "n_data_points": len(common_keys),
                "rates_a": dict(zip(frontends, [round(model_fe_rates[m1][fe], 4) for fe in frontends])),
                "rates_b": dict(zip(frontends, [round(model_fe_rates[m2][fe], 4) for fe in frontends])),
            })

    n_points = len(common_keys) if pairs else 0
    return {
        "pairs": pairs,
        "n_data_points": n_points,
        "n_models": len(models),
        "n_frontends": len(frontends),
        "valid": n_points >= 30,
        "note": f"Spearman valid ({n_points} points, threshold=30)" if n_points >= 30 else f"Only {n_points} points - need >=30",
    }


def _spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def _write_spearman(spearman: dict) -> None:
    json_path = PROCESSED_DIR / "phase2_spearman.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(spearman, f, indent=2, ensure_ascii=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SILP Phase 2 - Universality Benchmark Matrix")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--frontends", nargs="*", default=None)
    parser.add_argument("--judge", choices=["rule", "llm", "dual"], default="dual")
    parser.add_argument("--judge-model", default="glm-5.2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.models is None:
        args.models = ["deepseek-v3.2", "kimi-k2.6", "glm-5.2"]
    if args.frontends is None:
        from silp.frontend import list_frontends
        args.frontends = list_frontends()

    run_matrix(
        models=args.models,
        frontends=args.frontends,
        judge_mode=args.judge,
        judge_model=args.judge_model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
