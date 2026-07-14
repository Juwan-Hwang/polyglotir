#!/usr/bin/env python
"""Phase 3 — Heartbeat N-value experiment: cumulative error propagation in
stateful multi-turn sessions.

Per spec §2 Layer 3: "会话心跳每 N 条续期，N 值作阶段 3 实验变量"
Per spec §4 Phase 3: "心跳 N 值 vs 长会话累积错误传播找拐点"
Per spec §5: "鲁棒性（L1~L4 扰动曲线 + 上下文干扰识别 + 多轮累积错误传播）"

**Experiment design**:

In a stateful SILP session, the model sees all previous turns' payloads and
its own interpretations as conversation context. This context can pollute
subsequent turns — the model may conflate entities from previous payloads,
misinterpret the current payload by analogy to earlier ones, or drift in
its understanding of the SILP protocol.

The **heartbeat** mechanism (spec §2 Layer 3) renews the session every N
messages, clearing accumulated context. This experiment tests:

  - **N=1**: Stateless baseline — fresh context every turn (no accumulation)
  - **N=5,10,15,20**: Context accumulates for N turns, then heartbeat clears it
  - **N=∞**: No heartbeat — context accumulates indefinitely

**Key metrics**:
  - Per-turn success rate (does it degrade as context grows?)
  - Error propagation: P(fail at turn k+1 | fail at turn k) vs P(fail | pass)
  - Inflection point: the N where cumulative error rate starts climbing sharply

**Conversation simulation**:

The model receives a single prompt containing the accumulated context from
previous turns (payloads + interpretations), plus the current payload to
decode. When the heartbeat fires (every N turns), the accumulated context
is cleared — simulating session renewal.

**Data safety**:
  - Writes to ``data/raw/phase3_heartbeat/`` (NEW directory)
  - Results are **appended incrementally** — interrupted runs resume
  - **No error results recorded.** Infra errors retried indefinitely.

Usage::

    python scripts/run_heartbeat_n.py

    # If interrupted, re-run — completed (model, N, seed, turn) combos
    # are skipped automatically.
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
RAW_DIR = ROOT / "data" / "raw" / "phase3_heartbeat"

# ── Experiment parameters ─────────────────────────────────────────────

# N values: how many turns between heartbeats.
# N=1     → stateless (fresh context every turn)
# N=5..20 → context accumulates, heartbeat clears every N turns
# N=9999  → no heartbeat (infinite accumulation)
N_VALUES = [1, 5, 10, 15, 20, 9999]

# Turns per session (how many SILP payloads in one conversation)
TURNS_PER_SESSION = 15

# Repetitions per (model, N) — different case orderings for statistical power
# Expanded from 3 → 8 for adequate statistical power in error propagation
# and context-depth trend tests (seeds 0-2 already collected, 3-7 are new)
N_REPS = 8

# Models (subset of Phase 2's 5 for runtime efficiency)
MODELS = ["deepseek-v3.2", "glm-5.2", "kimi-k2.6"]

# Total runs: 3 models × 8 reps × 6 N values × 15 turns = 2160

# ── Prompt templates ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a SILP (Semantic Interlingua Layer Protocol) payload decoder. \
You are in a multi-turn session where multiple SILP payloads will be sent sequentially. \
For each payload, decode it and explain its full semantic intent including all \
conditions, entities, and alternatives."""

DECODE_PROMPT = """Decode the following SILP payload and explain what action(s) should be taken. \
Describe the full intent including all conditions, entities, and alternatives.

SILP payload:
{encoded}

Explain the semantic intent:"""

HEARTBEAT_MSG = "[SILP session renewed — previous context cleared]"


# ── Deterministic case ordering ───────────────────────────────────────


def _case_ordering(cases: list[tuple[str, object]], seed: int) -> list[tuple[str, object]]:
    """Deterministically shuffle cases for a given seed.

    Uses SHA256 to generate a permutation, ensuring:
    - Reproducibility (same seed = same ordering)
    - Different orderings across reps (different seeds)
    - No bias from Python's random module
    """
    n = len(cases)
    # Generate a sort key for each index based on SHA256
    indexed = list(range(n))
    indexed.sort(
        key=lambda i: hashlib.sha256(f"heartbeat_seed{seed}_idx{i}".encode()).hexdigest()
    )
    return [cases[i] for i in indexed]


# ── Context building ──────────────────────────────────────────────────


def _build_prompt(
    history: list[dict],
    current_encoded: str,
) -> str:
    """Build the full prompt with accumulated context.

    The history is a list of previous turns, each with:
        {"encoded": str, "response": str}

    The prompt includes:
    1. System prompt
    2. Previous turns (if any) as context
    3. Current payload to decode
    """
    parts = [SYSTEM_PROMPT]

    if history:
        parts.append("\n--- Previous turns (session context) ---\n")
        for i, h in enumerate(history):
            parts.append(f"[Turn {i+1}] SILP payload:\n{h['encoded']}\n")
            parts.append(f"[Turn {i+1}] Your interpretation:\n{h['response']}\n")

    parts.append("\n--- Current turn ---\n")
    parts.append(DECODE_PROMPT.format(encoded=current_encoded))
    return "\n".join(parts)


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

    Returns a set of ``"{model}|{N}|{seed}|{turn}"`` keys.
    """
    existing: set[str] = set()
    if not jsonl_path.exists():
        return existing
    for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            r = json.loads(line)
            key = f"{r['model']}|{r['n_value']}|{r['seed']}|{r['turn']}"
            existing.add(key)
        except (json.JSONDecodeError, KeyError):
            continue
    return existing


# ── Main experiment ───────────────────────────────────────────────────


def run_experiment() -> None:
    from silp.bench.models import GenerationConfig, get_model, load_env
    from silp.bench.judge import get_judge
    from silp.frontend import get_frontend

    load_env()

    cases = load_task_set()
    if not cases:
        print("Error: no valid IR cases found", file=sys.stderr)
        sys.exit(1)

    # Pre-compile all cases with code frontend
    fe_code = get_frontend("code")
    compiled: dict[str, tuple[object, str]] = {}  # case_id → (ir, encoded)
    for case_id, ir in cases:
        try:
            encoded = fe_code.compile(ir)
            compiled[case_id] = (ir, encoded)
        except Exception as exc:
            print(f"  [error] compile {case_id}: {exc}", file=sys.stderr)

    total = len(MODELS) * N_REPS * len(N_VALUES) * TURNS_PER_SESSION
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"SILP Phase 3 — Heartbeat N-value Experiment", file=sys.stderr)
    print(f"  Models:    {MODELS}", file=sys.stderr)
    print(f"  N values:  {N_VALUES}", file=sys.stderr)
    print(f"  Turns:     {TURNS_PER_SESSION}", file=sys.stderr)
    print(f"  Reps:      {N_REPS}", file=sys.stderr)
    print(f"  Total:     {total} runs", file=sys.stderr)
    print(f"  Output:    {RAW_DIR}", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    judge = get_judge("dual", "glm-5.2")

    run_count = 0
    skip_count = 0
    pass_count = 0

    for model_name in MODELS:
        model_slug = model_name.replace(".", "-").replace("/", "-")
        jsonl_path = RAW_DIR / model_slug / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        existing_keys = _load_existing(jsonl_path)
        if existing_keys:
            print(
                f"  [{model_name}] {len(existing_keys)} existing results, skipping",
                file=sys.stderr,
            )

        model = get_model(model_name)

        for seed in range(N_REPS):
            # Deterministic case ordering for this seed
            ordered_cases = _case_ordering(cases, seed)

            for n_val in N_VALUES:
                # ── Simulate one session ──────────────────────────
                history: list[dict] = []
                session_id = f"s{n_val}{seed:02d}"

                for turn in range(TURNS_PER_SESSION):
                    run_key = f"{model_name}|{n_val}|{seed}|{turn}"

                    if run_key in existing_keys:
                        skip_count += 1
                        # Still need to build history for subsequent turns
                        case_id = ordered_cases[turn % len(ordered_cases)][0]
                        ir_obj, encoded = compiled[case_id]
                        # We don't have the previous response, so we'll
                        # use a placeholder. This means interrupted runs
                        # that skip turns won't have perfect context, but
                        # the individual turn judgment is still valid.
                        history.append({
                            "encoded": encoded,
                            "response": "[previous response not available]",
                        })
                        # Heartbeat check
                        if n_val < 9999 and (turn + 1) % n_val == 0:
                            history = []
                        continue

                    case_id, ir_obj = ordered_cases[turn % len(ordered_cases)]
                    ir, encoded = compiled[case_id]

                    # Build prompt with accumulated context
                    prompt = _build_prompt(history, encoded)
                    run_count += 1
                    print(
                        f"  [{run_count}/{total - skip_count}] "
                        f"N={n_val} seed={seed} turn={turn} {case_id}",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

                    # ── Outer retry loop (no error results recorded) ──
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
                                    f"  retrying...",
                                    end="",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                continue
                            # Final attempt with extended timeout
                            print(
                                f"  final attempt...",
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
                            response = model.generate(
                                prompt,
                                GenerationConfig(
                                    max_new_tokens=256,
                                    temperature=0.0,
                                    timeout=60.0,
                                ),
                            )
                            if response.error:
                                print(
                                    f" -> UNRECOVERABLE after {outer_retries} retries",
                                    file=sys.stderr,
                                )
                            break
                        break

                    if response.error:
                        # SKIP — do NOT record error. Will retry on next run.
                        print(f" -> SKIPPED (will retry on next run)", file=sys.stderr)
                        run_count -= 1
                        continue

                    # Judge the response
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
                        "model": model_name,
                        "n_value": n_val,
                        "seed": seed,
                        "turn": turn,
                        "case_id": case_id,
                        "session_id": session_id,
                        "context_turns": len(history),
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

                    # Append to JSONL (incremental, never overwrite)
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    # Update history with the real response
                    history.append({
                        "encoded": encoded,
                        "response": response.text,
                    })

                    # ── Heartbeat: clear context every N turns ──
                    if n_val < 9999 and (turn + 1) % n_val == 0:
                        history = []

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Heartbeat N-value experiment complete!", file=sys.stderr)
    print(f"  New runs:   {run_count}", file=sys.stderr)
    print(f"  Skipped:    {skip_count} (already completed)", file=sys.stderr)
    print(f"  Passed:     {pass_count}", file=sys.stderr)
    if run_count:
        print(f"  Pass rate:  {pass_count / run_count * 100:.1f}%", file=sys.stderr)
    print(f"  Output:     {RAW_DIR}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)


if __name__ == "__main__":
    run_experiment()
