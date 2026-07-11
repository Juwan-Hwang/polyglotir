#!/usr/bin/env python
"""Phase 3 — Negation logic ablation: syntax skeleton vs vocabulary priors.

Per spec §4 Phase 3: "剥离三对照（实验组/乱序 token 组/语法骨架保留+词汇随机化）"

This experiment tests the core hypothesis: **shared syntax priors > shared
vocabulary priors**.  For each IR case, the code-frontend compilation is
transformed into three conditions:

1. **original** — the normal code-frontend output.
   e.g. ``if !rain(t+1): start(hike) else start(cards@indoor)``

2. **shuffled** — word-level tokens of the original, randomly permuted.
   Destroys syntactic structure but preserves the exact same vocabulary.
   e.g. ``cards@indoor else start(t+1) if !rain(hike): start``

3. **skeleton** — syntactic skeleton preserved, all semantic identifiers
   replaced with random meaningless tokens.
   e.g. ``if !aaa(t+1): bbb(ccc) else bbb(ddd@eee)``

If ``original >> shuffled`` and ``original ≈ skeleton``, syntax structure
is the primary driver (hypothesis confirmed).
If ``original >> skeleton``, vocabulary priors are essential (hypothesis
refuted).

Usage::

    # Full ablation: 3 conditions × 27 cases × N models
    python scripts/run_ablation.py --models deepseek-v3.2 glm-5.2

    # Dry run (show transformed encodings only)
    python scripts/run_ablation.py --dry-run

    # Specific cases only (e.g., negation cases)
    python scripts/run_ablation.py --filter case2
"""

from __future__ import annotations

import csv
import json
import random
import re
import string
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLES_DIR = ROOT / "examples"
RAW_DIR = ROOT / "data" / "raw" / "phase3_ablation"
PROCESSED_DIR = ROOT / "data" / "processed" / "phase3_ablation"

DECODE_PROMPT = """Decode the following SILP payload and explain what action(s) should be taken. Describe the full intent including all conditions, entities, and alternatives.

SILP payload:
{encoded}

Explain the semantic intent:"""

# ── Token classification for skeleton transformation ──────────────────

# Syntax keywords/symbols that are PRESERVED in the skeleton condition.
# Everything else is a "semantic" token and gets replaced.
_SYNTAX_KEYWORDS = frozenset({
    "if", "else", "and", "not", "True", "False",
})

# Single-char syntax symbols preserved as-is.
_SYNTAX_SYMBOLS = frozenset("!():;,=@<>")

# Regex to identify "semantic" tokens: any maximal run of characters
# that are word chars, dots, hyphens, underscores, plus signs (covers
# ``hike``, ``t+1``, ``5h``, ``4.5``, ``open_now``, ``fr_rev_bold``).
# Syntax keywords (if/else/and) are excluded *after* matching.
_RE_SEMANTIC = re.compile(r"[\w.+-]+")


def _generate_placeholder_map(tokens: list[str], seed: int = 42) -> dict[str, str]:
    """Map each unique semantic token to a deterministic random placeholder.

    Uses a seeded RNG so the same input always produces the same mapping
    (reproducibility).  Placeholders are 3-letter lowercase strings from
    the set [a-z], giving 26³ = 17,576 unique values — far more than needed.

    >>> m = _generate_placeholder_map(["rain", "hike", "cards"])
    >>> all(v != k for k, v in m.items())
    True
    >>> len(set(m.values())) == len(m)
    True
    """
    rng = random.Random(seed)
    pool = list(string.ascii_lowercase)
    used: set[str] = set()
    mapping: dict[str, str] = {}

    for tok in sorted(set(tokens)):  # deterministic iteration
        while True:
            placeholder = "".join(rng.choice(pool) for _ in range(3))
            if placeholder not in used:
                used.add(placeholder)
                mapping[tok] = placeholder
                break

    return mapping


def make_shuffled(original: str, seed: int = 42) -> str:
    """Shuffle word-level tokens of *original*, preserving the exact
    vocabulary but destroying syntactic structure.

    Splits on whitespace, shuffles, rejoins.
    """
    tokens = original.split()
    rng = random.Random(seed)
    rng.shuffle(tokens)
    return " ".join(tokens)


def make_skeleton(original: str, seed: int = 42) -> str:
    """Replace all semantic identifiers with random placeholders while
    preserving the syntactic skeleton.

    Rules:
    - Syntax keywords (``if``, ``else``, ``and``) → preserved
    - Pure punctuation / operators → preserved
    - Everything else (function names, arguments, constraint types,
      values like ``hike``, ``Beijing``, ``t+1``) → replaced with a
      deterministic 3-letter placeholder

    ``t+1`` and similar time expressions are NOT preserved — the model
    must rely on syntactic position, not the familiar ``t+`` pattern.

    >>> make_skeleton("if !rain(t+1): start(hike) else start(cards@indoor)")
    'if !aaa(bbb): ccc(ddd) else ccc(eee@fff)'
    """
    # Collect all semantic tokens for the placeholder map.
    semantic_tokens = [
        m.group()
        for m in _RE_SEMANTIC.finditer(original)
        if m.group().lower() not in _SYNTAX_KEYWORDS
    ]
    pmap = _generate_placeholder_map(semantic_tokens, seed=seed)

    def _replace(m: re.Match[str]) -> str:
        tok = m.group()
        if tok.lower() in _SYNTAX_KEYWORDS:
            return tok
        return pmap.get(tok, tok)

    return _RE_SEMANTIC.sub(_replace, original)


# ── Task loading (reused from Phase 2) ────────────────────────────────


def load_task_set(filter_pattern: str | None = None) -> list[tuple[str, object]]:
    """Load all example IR files, optionally filtered by name pattern."""
    from silp.ir import validate as validate_ir

    cases = []
    for path in sorted(EXAMPLES_DIR.glob("case*.json")):
        if filter_pattern and filter_pattern not in path.stem:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result = validate_ir(data)
        if not result.valid:
            print(f"  [skip] {path.name}: {result.errors}", file=sys.stderr)
            continue
        cases.append((path.stem, result.ir))
    return cases


# ── Main experiment ───────────────────────────────────────────────────

CONDITIONS = ("original", "shuffled", "skeleton")


def run_ablation(
    models: list[str],
    judge_mode: str = "dual",
    judge_model: str = "glm-5.2",
    filter_pattern: str | None = None,
    dry_run: bool = False,
) -> None:
    from silp.bench.models import GenerationConfig, get_model, list_models, load_env
    from silp.bench.judge import get_judge
    from silp.frontend.code import CodeFrontend

    load_env()

    # Validate model names
    available = list_models()
    for m in models:
        if m not in available:
            print(f"Error: unknown model {m!r}", file=sys.stderr)
            sys.exit(1)

    cases = load_task_set(filter_pattern)
    if not cases:
        print("Error: no valid IR cases found", file=sys.stderr)
        sys.exit(1)

    code_fe = CodeFrontend()

    # Pre-compile all cases and generate three conditions
    compiled: dict[tuple[str, str], str] = {}  # (case_id, condition) -> encoded
    for case_id, ir in cases:
        original = code_fe.compile(ir)
        compiled[(case_id, "original")] = original
        # Deterministic seed from case_id (PYTHONHASHSAFE)
        _seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
        compiled[(case_id, "shuffled")] = make_shuffled(original, seed=_seed)
        compiled[(case_id, "skeleton")] = make_skeleton(original, seed=_seed)

    total = len(cases) * len(CONDITIONS) * len(models)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"SILP Phase 3 — Ablation: Syntax vs Vocabulary Priors", file=sys.stderr)
    print(f"  Cases:      {len(cases)}", file=sys.stderr)
    print(f"  Conditions: {list(CONDITIONS)}", file=sys.stderr)
    print(f"  Models:     {models}", file=sys.stderr)
    print(f"  Judge:      {judge_mode}" + (f" ({judge_model})" if judge_mode in ("llm", "dual") else ""), file=sys.stderr)
    print(f"  Total:      {total} runs", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    if dry_run:
        print("Dry run — showing transformations:\n", file=sys.stderr)
        for case_id, ir in cases:
            print(f"  [{case_id}]", file=sys.stderr)
            for cond in CONDITIONS:
                print(f"    {cond:10s}: {compiled[(case_id, cond)]}", file=sys.stderr)
            print(file=sys.stderr)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    judge = get_judge(judge_mode, judge_model)

    all_results: list[dict[str, object]] = []
    pass_counts: dict[str, int] = {c: 0 for c in CONDITIONS}
    total_counts: dict[str, int] = {c: 0 for c in CONDITIONS}

    for model_name in models:
        model_slug = model_name.replace(".", "-").replace("/", "-")
        jsonl_path = RAW_DIR / model_slug / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

        model = get_model(model_name)

        for case_id, ir in cases:
            for condition in CONDITIONS:
                encoded = compiled[(case_id, condition)]
                run_id = f"{case_id}|{condition}|{model_name}"
                idx = len(all_results) + 1
                print(f"  [{idx}/{total}] {run_id}", end="", file=sys.stderr, flush=True)

                prompt = DECODE_PROMPT.format(encoded=encoded)
                response = model.generate(
                    prompt,
                    GenerationConfig(max_new_tokens=256, temperature=0.0, timeout=30.0),
                )

                if response.error:
                    print(f" -> ERROR: {response.error[:60]}", file=sys.stderr)
                    result = {
                        "case_id": case_id, "condition": condition,
                        "model": model_name, "encoded": encoded,
                        "model_response": "", "judge_verdict": "fail",
                        "judge_reason": f"Model error: {response.error}",
                        "judge": "error", "elapsed": response.elapsed,
                        "retries": response.retries,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": False,
                    }
                else:
                    judge_result = judge.judge(ir, encoded, response.text)
                    if judge_result.passed:
                        pass_counts[condition] += 1
                        print(f" -> PASS ({response.elapsed:.1f}s)", file=sys.stderr)
                    else:
                        print(f" -> FAIL ({response.elapsed:.1f}s): {judge_result.reason[:60]}", file=sys.stderr)

                    details = judge_result.details
                    result = {
                        "case_id": case_id, "condition": condition,
                        "model": model_name, "encoded": encoded,
                        "model_response": response.text,
                        "judge_verdict": judge_result.verdict,
                        "judge_reason": judge_result.reason,
                        "judge": judge_result.judge,
                        "rule_verdict": details.get("rule_verdict"),
                        "rule_reason": details.get("rule_reason"),
                        "llm_verdict": details.get("llm_verdict"),
                        "llm_reason": details.get("llm_reason"),
                        "elapsed": response.elapsed,
                        "retries": response.retries,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "first_pass": judge_result.passed,
                    }

                total_counts[condition] += 1
                all_results.append(result)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Ablation experiment complete!", file=sys.stderr)
    print(f"\n  Pass rates by condition:", file=sys.stderr)
    for cond in CONDITIONS:
        rate = pass_counts[cond] / total_counts[cond] * 100 if total_counts[cond] else 0
        print(f"    {cond:10s}: {pass_counts[cond]:3d}/{total_counts[cond]:3d} ({rate:5.1f}%)", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    _write_ablation_csv(all_results, models)
    _write_ablation_matrix(all_results, models)


def _write_ablation_matrix(results: list[dict], models: list[str]) -> None:
    """Write the condition × model pass-rate matrix."""
    csv_path = PROCESSED_DIR / "ablation_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition"] + models)
        for cond in CONDITIONS:
            row = [cond]
            for model in models:
                subset = [r for r in results
                          if r["condition"] == cond and r["model"] == model]
                if subset:
                    rate = sum(1 for r in subset if r["first_pass"]) / len(subset) * 100
                    row.append(f"{rate:.1f}%")
                else:
                    row.append("N/A")
            writer.writerow(row)
    print(f"  Matrix: {csv_path}", file=sys.stderr)

    # Also write per-case detail
    csv_path = PROCESSED_DIR / "ablation_case_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "condition", "model", "first_pass",
            "judge_verdict", "judge_reason", "elapsed",
        ])
        for r in results:
            writer.writerow([
                r["case_id"], r["condition"], r["model"],
                r["first_pass"], r["judge_verdict"],
                str(r["judge_reason"])[:120],
                f"{r['elapsed']:.1f}",
            ])
    print(f"  Details: {csv_path}", file=sys.stderr)


def _write_ablation_csv(results: list[dict], models: list[str]) -> None:
    """Write the full per-case results for each model."""
    for model in models:
        model_slug = model.replace(".", "-").replace("/", "-")
        csv_path = PROCESSED_DIR / f"ablation_{model_slug}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "condition", "first_pass", "elapsed", "judge"])
            for r in results:
                if r["model"] != model:
                    continue
                writer.writerow([
                    r["case_id"], r["condition"],
                    r["first_pass"], f"{r['elapsed']:.1f}",
                    r["judge"],
                ])


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 ablation: syntax skeleton vs vocabulary priors",
    )
    parser.add_argument(
        "--models", nargs="+", default=["deepseek-v3.2"],
        help="Model names to test",
    )
    parser.add_argument(
        "--judge", default="dual", choices=["rule", "llm", "dual"],
        help="Judge mode",
    )
    parser.add_argument(
        "--judge-model", default="glm-5.2",
        help="Model for LLM judge",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Only run cases matching this pattern (e.g., 'case2' for negation)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show transformed encodings without calling models",
    )

    args = parser.parse_args()
    run_ablation(
        models=args.models,
        judge_mode=args.judge,
        judge_model=args.judge_model,
        filter_pattern=args.filter,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
