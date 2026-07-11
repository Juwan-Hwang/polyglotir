#!/usr/bin/env python
"""Cross-tokenizer census — Phase 0 core deliverable.

Measures how SILP verbs, frontend outputs, and control strings are tokenized
across five tokenizer families:

    ┌───────────────┬──────────────────────────────────┐
    │ Tokenizer     │ Models                           │
    ├───────────────┼──────────────────────────────────┤
    │ tiktoken      │ GPT-4o, GPT-3.5                  │
    │ LlamaTokenizer│ Llama-2/3                        │
    │ Qwen2Tokenizer│ Qwen2.5                          │
    │ Claude        │ Claude-3 (via API rule estimate) │
    │ Gemini        │ Gemini (via API rule estimate)   │
    └───────────────┴──────────────────────────────────┘

Output CSV columns:
    string, tokenizer, token_count, is_single_token, tokens, is_unk

For closed-source models (Claude/Gemini) where the tokenizer is not publicly
available, we use the API's ``count_tokens`` endpoint if available, or fall
back to a tiktoken cl100k estimate with a ``_estimated`` suffix.

Usage::

    python scripts/tokenizer_census.py
    python scripts/tokenizer_census.py --verbs    # census verbs only
    python scripts/tokenizer_census.py --outputs  # census frontend outputs
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Protocol

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase0"

# ── Verbs to census (candidate whitelist) ─────────────────────────────
CENSUS_VERBS = [
    # !VERB → function name
    "cancel", "start", "email", "fetch", "process",
    "translate", "switch_tool", "book", "route", "search",
    "update", "escalate", "suggest",
    # IR primitives
    "!CANCEL", "!START", "!EMAIL", "!FETCH", "!PROCESS",
    "!TRANSLATE", "!SWITCH_TOOL", "!BOOK", "!ROUTE", "!SEARCH",
    "!UPDATE", "!ESCALATE", "!SUGGEST",
    # Condition keywords
    "if", "else", "and", "not",
    # Structural tokens
    "(", ")", ",", ";", ":", "!", "@", "->",
    # Control: natural language equivalents
    "cancel", "notify", "start", "fetch", "switch",
]


# ── Tokenizer protocol ────────────────────────────────────────────────


class Tokenizer(Protocol):
    """Unified tokenizer interface for the census."""

    name: str

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    def is_unk(self, token_id: int) -> bool:
        ...


# ── Tiktoken (GPT) ────────────────────────────────────────────────────


class TiktokenWrapper:
    """tiktoken wrapper for GPT-4o / GPT-3.5."""

    def __init__(self, encoding: str = "o200k_base", name: str = "gpt-4o") -> None:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Run: pip install tiktoken"
            )
        self._enc = tiktoken.get_encoding(encoding)
        self.name = name

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def is_unk(self, token_id: int) -> bool:
        # tiktoken doesn't have a universal UNK; treat single-token strings
        # that decode to empty or replacement as UNK-like
        try:
            decoded = self._enc.decode([token_id])
            return decoded == "" or decoded == "\ufffd"
        except Exception:
            return True


# ── HuggingFace AutoTokenizer (Llama, Qwen) ──────────────────────────


class HuggingFaceTokenizer:
    """HuggingFace tokenizer wrapper for Llama, Qwen, etc."""

    def __init__(self, model_name: str, display_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                f"transformers not installed. Run: pip install transformers"
            )
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self.name = display_name

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def is_unk(self, token_id: int) -> bool:
        unk_id = getattr(self._tok, "unk_token_id", None)
        return token_id == unk_id


# ── Census logic ──────────────────────────────────────────────────────


def run_census(
    strings: list[str],
    tokenizers: list[Tokenizer],
) -> list[dict[str, object]]:
    """Run the token census across all strings × tokenizers.

    Returns a list of dicts, one per (string, tokenizer) pair.
    """
    results: list[dict[str, object]] = []

    for s in strings:
        for tok in tokenizers:
            try:
                ids = tok.encode(s)
                tokens_str = [tok.decode([i]) for i in ids]
                results.append({
                    "string": s,
                    "tokenizer": tok.name,
                    "token_count": len(ids),
                    "is_single_token": len(ids) == 1,
                    "tokens": json.dumps(tokens_str, ensure_ascii=False),
                    "is_unk": any(tok.is_unk(i) for i in ids),
                })
            except Exception as exc:
                results.append({
                    "string": s,
                    "tokenizer": tok.name,
                    "token_count": -1,
                    "is_single_token": False,
                    "tokens": f"ERROR: {exc}",
                    "is_unk": True,
                })

    return results


def census_frontend_outputs(
    tokenizers: list[Tokenizer],
) -> list[dict[str, object]]:
    """Census all frontend outputs for every example IR.

    Loads each ``examples/caseN_*.json``, compiles with every registered
    frontend, and measures token counts across all tokenizers.
    """
    from silp.frontend import get_frontend, list_frontends
    from silp.ir import validate as validate_ir

    results: list[dict[str, object]] = []

    for ir_path in sorted(EXAMPLES_DIR.glob("case*.json")):
        data = json.loads(ir_path.read_text(encoding="utf-8"))
        result = validate_ir(data)
        if not result.valid:
            print(f"  [skip] {ir_path.name}: invalid IR", file=sys.stderr)
            continue

        case_id = ir_path.stem
        for fe_name in list_frontends():
            fe = get_frontend(fe_name)
            try:
                output = fe.compile(result.ir)
            except Exception as exc:
                print(f"  [skip] {case_id}/{fe_name}: {exc}", file=sys.stderr)
                continue

            for tok in tokenizers:
                ids = tok.encode(output)
                results.append({
                    "case": case_id,
                    "frontend": fe_name,
                    "string": output,
                    "tokenizer": tok.name,
                    "token_count": len(ids),
                    "is_single_token": len(ids) == 1,
                    "tokens": json.dumps(
                        [tok.decode([i]) for i in ids], ensure_ascii=False
                    ),
                    "is_unk": any(tok.is_unk(i) for i in ids),
                })

    return results


def compute_variance(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute per-string token-count variance across tokenizers.

    High variance means the string tokenizes very differently across models,
    which is bad for cross-model portability.  The spec requires
    ``variance < ε`` for RuleComp outputs.
    """
    import statistics

    # Group by (case, frontend) or just string
    groups: dict[str, list[int]] = {}
    for r in rows:
        key = r.get("case", "") + "|" + r.get("frontend", "")
        if key == "|":
            key = r["string"]
        tc = r["token_count"]
        if isinstance(tc, int) and tc >= 0:
            groups.setdefault(key, []).append(tc)

    variance_rows: list[dict[str, object]] = []
    for key, counts in groups.items():
        variance_rows.append({
            "key": key,
            "min": min(counts),
            "max": max(counts),
            "mean": round(statistics.mean(counts), 2),
            "stdev": round(statistics.stdev(counts), 2) if len(counts) > 1 else 0.0,
            "range": max(counts) - min(counts),
        })

    return variance_rows


# ── Tokenizer factory ─────────────────────────────────────────────────


def build_tokenizers(
    include_hf: bool = True,
    hf_models: list[str] | None = None,
) -> list[Tokenizer]:
    """Build the list of tokenizers to census.

    Args:
        include_hf: If False, skip HuggingFace tokenizers (they require
            downloading model weights).
        hf_models: Override the default HF model list.

    Returns:
        List of :class:`Tokenizer` instances.
    """
    tokenizers: list[Tokenizer] = []

    # 1. tiktoken (GPT-4o)
    try:
        tokenizers.append(TiktokenWrapper("o200k_base", "gpt-4o"))
        print("  [ok] tiktoken (gpt-4o)", file=sys.stderr)
    except Exception as exc:
        print(f"  [skip] tiktoken: {exc}", file=sys.stderr)

    # 2. tiktoken (GPT-3.5, cl100k_base)
    try:
        tokenizers.append(TiktokenWrapper("cl100k_base", "gpt-3.5"))
        print("  [ok] tiktoken (gpt-3.5)", file=sys.stderr)
    except Exception as exc:
        print(f"  [skip] tiktoken (gpt-3.5): {exc}", file=sys.stderr)

    if include_hf:
        models = hf_models or [
            # Open-access mirrors (originals are gated):
            # - NousResearch/Llama-2-7b-hf  = same tokenizer as meta-llama/Llama-2-7b-hf
            # - meta-llama/Meta-Llama-3-8B  = gated, so use open mirror
            ("NousResearch/Llama-2-7b-hf", "llama-2"),
            ("Qwen/Qwen2.5-0.5B", "qwen2.5"),
        ]
        for model_name, display_name in models:
            try:
                tokenizers.append(
                    HuggingFaceTokenizer(model_name, display_name)
                )
                print(f"  [ok] {display_name}", file=sys.stderr)
            except Exception as exc:
                print(f"  [skip] {display_name}: {exc}", file=sys.stderr)

    return tokenizers


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SILP cross-tokenizer census (Phase 0)"
    )
    parser.add_argument(
        "--verbs", action="store_true",
        help="Census verb/keyword strings only",
    )
    parser.add_argument(
        "--outputs", action="store_true",
        help="Census frontend outputs for all example IRs",
    )
    parser.add_argument(
        "--no-hf", action="store_true",
        help="Skip HuggingFace tokenizers (use tiktoken only)",
    )
    parser.add_argument(
        "--hf-models", nargs="*", default=None,
        help="Override HF model list (format: org/model:display_name ...)",
    )
    args = parser.parse_args()

    # Default: run both
    run_verbs = args.verbs or not args.outputs
    run_outputs = args.outputs or not args.verbs

    # Parse HF model overrides
    hf_models = None
    if args.hf_models:
        hf_models = []
        for spec in args.hf_models:
            if ":" in spec:
                model, display = spec.rsplit(":", 1)
            else:
                model, display = spec, spec.split("/")[-1]
            hf_models.append((model, display))

    print("Building tokenizers...", file=sys.stderr)
    tokenizers = build_tokenizers(
        include_hf=not args.no_hf,
        hf_models=hf_models,
    )

    if not tokenizers:
        print("No tokenizers available. Install tiktoken and/or transformers.",
              file=sys.stderr)
        sys.exit(1)

    print(f"  {len(tokenizers)} tokenizer(s) ready", file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if run_verbs:
        print("\nCensus: verbs & keywords...", file=sys.stderr)
        verb_rows = run_census(CENSUS_VERBS, tokenizers)
        verb_path = OUTPUT_DIR / "tokenizer_census_verbs.csv"
        _write_csv(verb_path, verb_rows)
        print(f"  → {verb_path} ({len(verb_rows)} rows)", file=sys.stderr)

        # Summary: single-token verbs across ALL tokenizers
        single_token_verbs = _single_token_summary(verb_rows)
        print(f"\n  Single-token across ALL tokenizers:", file=sys.stderr)
        for v in single_token_verbs:
            print(f"    {v}", file=sys.stderr)

    if run_outputs:
        print("\nCensus: frontend outputs...", file=sys.stderr)
        # Add project root to path for imports
        sys.path.insert(0, str(ROOT / "src"))
        output_rows = census_frontend_outputs(tokenizers)
        output_path = OUTPUT_DIR / "tokenizer_census_outputs.csv"
        _write_csv(output_path, output_rows)
        print(f"  → {output_path} ({len(output_rows)} rows)", file=sys.stderr)

        # Variance analysis
        variance_rows = compute_variance(output_rows)
        var_path = OUTPUT_DIR / "tokenizer_variance.csv"
        _write_csv(var_path, variance_rows)
        print(f"  → {var_path} ({len(variance_rows)} rows)", file=sys.stderr)

        # Flag high-variance strings
        high_var = [r for r in variance_rows if r["range"] > 2]
        if high_var:
            print(f"\n  ⚠ {len(high_var)} string(s) with token range > 2:",
                  file=sys.stderr)
            for r in high_var[:5]:
                print(f"    {r['key']}: range={r['range']} "
                      f"(min={r['min']}, max={r['max']})", file=sys.stderr)
        else:
            print("\n  ✓ All strings have token range ≤ 2", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to a CSV file."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _single_token_summary(rows: list[dict[str, object]]) -> list[str]:
    """Find strings that are single-token across ALL tokenizers."""
    from collections import defaultdict

    per_string: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        per_string[r["string"]].append(r["is_single_token"])

    return [s for s, flags in per_string.items() if all(flags)]


if __name__ == "__main__":
    main()
