#!/usr/bin/env python
"""Phase 3 — Entropy sweep: progressive compression on the code frontend.

Generates 7 compression levels (0%, 15%, 30%, 45%, 60%, 75%, 90% character
deletion) to test whether success rate varies continuously with compression-
based entropy, or whether there's a sharp inflection point ("cliff").

The deletion is **deterministic** (seeded by case_id) — reproducible across
runs and independent of model or judge state.

**Data safety**:
  - Writes to ``data/raw/phase3_entropy_sweep/`` (NEW directory — does NOT
    touch Phase 2 data).
  - Results are **appended incrementally** — interrupted runs resume without
    data loss. The output file is NEVER cleared.
  - **No error results are ever recorded.** Infra errors (401, timeout,
    empty response) are retried indefinitely (outer loop: 10 rounds, then
    extended timeout, same policy as Phase 2). If a run truly cannot
    complete, it is skipped (not recorded) and will be retried on the
    next execution.

Usage::

    python scripts/run_entropy_sweep.py

    # If interrupted, just re-run — completed runs are skipped automatically.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLES_DIR = ROOT / "examples"
RAW_DIR = ROOT / "data" / "raw" / "phase3_entropy_sweep"

# 7 compression levels: 0% = full code frontend, 90% = only 10% of chars remain
DELETION_RATES = [0, 15, 30, 45, 60, 75, 90]

# Use 3 models (subset of Phase 2's 5) for runtime efficiency
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6"]

DECODE_PROMPT = """Decode the following SILP payload and explain what action(s) should be taken. Describe the full intent including all conditions, entities, and alternatives.

SILP payload:
{encoded}

Explain the semantic intent:"""


# ── Progressive compression ───────────────────────────────────────────


def progressive_compress(text: str, deletion_pct: int, seed_str: str) -> str:
    """Delete a proportion of characters deterministically.

    Uses SHA256 seeded by ``seed_str`` (case_id) to select which characters
    to keep. This ensures:
    - **Reproducibility**: same case_id + same rate = identical output.
    - **Fairness**: every character has equal probability of deletion.
    - **Independence**: deletion pattern is uncorrelated with content.

    At ``deletion_pct=0`` the original text is returned unchanged, providing
    a baseline identical to the Phase 2 code frontend.

    Args:
        text: Original encoded text.
        deletion_pct: Percentage of characters to delete (0–100).
        seed_str: Deterministic seed (typically the case_id).

    Returns:
        Compressed text with the specified proportion of characters removed.
    """
    if deletion_pct <= 0 or not text:
        return text
    n = len(text)
    n_keep = int(n * (100 - deletion_pct) / 100)
    if n_keep >= n:
        return text
    # Deterministic permutation: sort indices by their SHA256 hash
    indices = list(range(n))
    indices.sort(
        key=lambda i: hashlib.sha256(f"{seed_str}:{i}".encode()).hexdigest()
    )
    keep_set = set(indices[:n_keep])
    return "".join(text[i] for i in range(n) if i in keep_set)


# ── Data loading ──────────────────────────────────────────────────────


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


def _load_existing(jsonl_path: Path) -> set[str]:
    """Load existing result keys to skip completed runs.

    Returns a set of ``"{case_id}|{frontend}|{model}"`` keys for runs
    that are already in the JSONL file. This enables incremental append
    without data loss or duplication.
    """
    existing: set[str] = set()
    if not jsonl_path.exists():
        return existing
    for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            r = json.loads(line)
            key = f"{r['case_id']}|{r['frontend']}|{r['model']}"
            existing.add(key)
        except (json.JSONDecodeError, KeyError):
            continue
    return existing


# ── Main sweep ────────────────────────────────────────────────────────


def run_sweep() -> None:
    from silp.bench.models import GenerationConfig, get_model, load_env
    from silp.bench.judge import get_judge
    from silp.frontend import get_frontend

    load_env()

    cases = load_task_set()
    if not cases:
        print("Error: no valid IR cases found", file=sys.stderr)
        sys.exit(1)

    # ── Pre-compile all IRs with the code frontend ─────────────────
    fe_code = get_frontend("code")
    full_encoded: dict[str, str] = {}
    for case_id, ir in cases:
        try:
            full_encoded[case_id] = fe_code.compile(ir)
        except Exception as exc:
            print(f"  [error] compile {case_id}: {exc}", file=sys.stderr)
            full_encoded[case_id] = ""

    # ── Generate compressed versions for each deletion rate ───────
    compressed: dict[tuple[str, int], str] = {}
    for case_id, text in full_encoded.items():
        for rate in DELETION_RATES:
            compressed[(case_id, rate)] = progressive_compress(text, rate, case_id)

    total = len(cases) * len(DELETION_RATES) * len(MODELS)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"SILP Phase 3 - Entropy Sweep (Progressive Compression)", file=sys.stderr)
    print(f"  Cases:     {len(cases)}", file=sys.stderr)
    print(f"  Levels:    {DELETION_RATES}", file=sys.stderr)
    print(f"  Models:    {MODELS}", file=sys.stderr)
    print(f"  Total:     {total} runs", file=sys.stderr)
    print(f"  Output:    {RAW_DIR}", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    judge = get_judge("dual", "glm-5.2")

    pass_count = 0
    run_count = 0
    skip_count = 0

    for model_name in MODELS:
        model_slug = model_name.replace(".", "-").replace("/", "-")
        jsonl_path = RAW_DIR / model_slug / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results to skip completed runs (incremental append)
        existing_keys = _load_existing(jsonl_path)
        if existing_keys:
            print(
                f"  [{model_name}] {len(existing_keys)} existing results, skipping",
                file=sys.stderr,
            )

        model = get_model(model_name)

        for case_id, ir in cases:
            for rate in DELETION_RATES:
                fe_name = f"code_del{rate}"
                run_key = f"{case_id}|{fe_name}|{model_name}"

                if run_key in existing_keys:
                    skip_count += 1
                    continue

                encoded = compressed.get((case_id, rate), "")
                if not encoded:
                    continue

                prompt = DECODE_PROMPT.format(encoded=encoded)
                run_count += 1
                print(
                    f"  [{run_count}/{total - skip_count}] {run_key}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

                # ── Outer retry loop (same as Phase 2) ──────────────
                # Keep retrying on infra errors until we get a real response.
                # No error results are ever recorded.
                outer_retries = 0
                response = None
                while True:
                    response = model.generate(
                        prompt,
                        GenerationConfig(
                            max_new_tokens=256, temperature=0.0, timeout=30.0
                        ),
                    )
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
                            print(
                                f"  retrying {run_key}...",
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        # Final attempt with extended timeout
                        print(
                            f"  final attempt with extended timeout...",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                        response = model.generate(
                            prompt,
                            GenerationConfig(
                                max_new_tokens=256, temperature=0.0, timeout=60.0
                            ),
                        )
                        if response.error:
                            print(
                                f" -> UNRECOVERABLE ERROR after {outer_retries} retries",
                                file=sys.stderr,
                            )
                        break
                    break

                if response.error:
                    # SKIP this run — do NOT record an error result.
                    # It will be retried on the next execution.
                    print(
                        f" -> SKIPPED (will retry on next run)",
                        file=sys.stderr,
                    )
                    run_count -= 1  # don't count skipped runs
                    continue

                judge_result = judge.judge(ir, encoded, response.text)
                if judge_result.passed:
                    pass_count += 1
                    print(f" -> PASS ({response.elapsed:.1f}s)", file=sys.stderr)
                else:
                    print(
                        f" -> FAIL ({response.elapsed:.1f}s): "
                        f"{judge_result.reason[:60]}",
                        file=sys.stderr,
                    )

                details = judge_result.details
                result = {
                    "case_id": case_id,
                    "frontend": fe_name,
                    "model": model_name,
                    "deletion_rate": rate,
                    "encoded": encoded,
                    "model_response": response.text,
                    "judge_verdict": judge_result.verdict,
                    "judge_reason": judge_result.reason,
                    "judge": judge_result.judge,
                    "rule_verdict": details.get("rule_verdict"),
                    "rule_reason": details.get("rule_reason"),
                    "llm_verdict": details.get("llm_verdict"),
                    "llm_reason": details.get("llm_reason"),
                    "elapsed": response.elapsed,
                    "retries": response.retries + outer_retries,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "first_pass": judge_result.passed,
                }

                # Append to JSONL (do NOT overwrite existing data)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Entropy sweep complete!", file=sys.stderr)
    print(f"  New runs:      {run_count}", file=sys.stderr)
    print(f"  Skipped:       {skip_count} (already completed)", file=sys.stderr)
    print(f"  Passed:        {pass_count}", file=sys.stderr)
    if run_count:
        print(f"  Pass rate:     {pass_count / run_count * 100:.1f}%", file=sys.stderr)
    print(f"  Output:        {RAW_DIR}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)


if __name__ == "__main__":
    run_sweep()
