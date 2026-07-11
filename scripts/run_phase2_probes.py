#!/usr/bin/env python
"""Phase 2 probes — zero-width/homoglyph detection + trigger-head truncation.

Per spec section 4 Phase 2:
    "加零宽/同形探针、触发头分词截断、分词差异定量阈值实验"

Three experiments:

1. **Zero-width / homoglyph probe** (spec section 7: "零宽/同形仅作无载荷探针"):
   Inject zero-width characters (U+200B, U+200C, U+200D, U+FEFF) and
   Unicode homoglyphs (Cyrillic 'а' vs Latin 'a') into SILP payloads,
   then measure how each tokenizer handles them. This is a *probe* —
   no hidden payload, just detection of whether these characters survive
   tokenization intact or get silently mangled.

2. **Trigger-head tokenization truncation** (spec section 3: "触发头过分词器兼容矩阵"):
   Test how the SILP trigger header (``//silp:v1``) and alternative
   headers (``$``, ``//``, ``##``) tokenize across tokenizer families.
   Measure whether the header gets split into fragments that could break
   auto-detection. Also test 1-character version numbers.

3. **Token-count variance threshold** (spec section 4 Phase 0: "token 数方差 < epsilon"):
   For each frontend output, measure the max token-count range across
   all tokenizers. Flag any string where range > threshold (default 3).

Output:
    data/processed/phase2/probe_zerowidth.csv
    data/processed/phase2/probe_trigger_head.csv
    data/processed/phase2/probe_variance_threshold.csv

Usage::

    python scripts/run_phase2_probes.py
    python scripts/run_phase2_probes.py --no-hf   # tiktoken only
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "data" / "processed" / "phase2"

# Reuse the tokenizer infrastructure from Phase 0 census
sys.path.insert(0, str(ROOT / "scripts"))
from tokenizer_census import build_tokenizers, run_census  # type: ignore[import-not-found]


# ── 1. Zero-width / homoglyph probe ───────────────────────────────────

# Zero-width and invisible characters
ZERO_WIDTH_CHARS = [
    ("\u200b", "ZWSP", "Zero-Width Space"),
    ("\u200c", "ZWNJ", "Zero-Width Non-Joiner"),
    ("\u200d", "ZWJ", "Zero-Width Joiner"),
    ("\ufeff", "BOM", "Byte Order Mark / ZWNBSP"),
]

# Homoglyph pairs: (Latin, Cyrillic/other, description)
HOMOGLYPH_PAIRS = [
    ("a", "\u0430", "Latin-a vs Cyrillic-a"),
    ("e", "\u0435", "Latin-e vs Cyrillic-e"),
    ("o", "\u043e", "Latin-o vs Cyrillic-o"),
    ("p", "\u0440", "Latin-p vs Cyrillic-r"),
    ("c", "\u0441", "Latin-c vs Cyrillic-es"),
    ("x", "\u0445", "Latin-x vs Cyrillic-ha"),
    ("y", "\u0443", "Latin-y vs Cyrillic-u"),
]

# SILP verbs to test with injected zero-width chars
PROBE_VERBS = ["cancel", "start", "translate", "if", "else"]


def probe_zerowidth(tokenizers) -> list[dict[str, object]]:
    """Test how zero-width characters and homoglyphs tokenize.

    For each probe verb, inject a zero-width char in the middle and
    measure whether the token count changes (indicating the invisible
    char broke a single token into fragments).
    """
    results: list[dict[str, object]] = []

    # Test 1: Zero-width chars injected into verbs
    for verb in PROBE_VERBS:
        # Baseline: clean verb
        clean_ids = _encode_all(verb, tokenizers)

        for char, code, desc in ZERO_WIDTH_CHARS:
            # Inject in the middle of the verb
            mid = len(verb) // 2
            injected = verb[:mid] + char + verb[mid:]

            for tok in tokenizers:
                try:
                    ids = tok.encode(injected)
                    clean = clean_ids.get(tok.name, [])
                    results.append({
                        "test": "zerowidth_inject",
                        "verb": verb,
                        "char_code": code,
                        "char_desc": desc,
                        "tokenizer": tok.name,
                        "clean_token_count": len(clean),
                        "injected_token_count": len(ids),
                        "delta": len(ids) - len(clean),
                        "survived": _char_survives(tok, char),
                        "injected_string_repr": repr(injected),
                    })
                except Exception as exc:
                    results.append({
                        "test": "zerowidth_inject",
                        "verb": verb,
                        "char_code": code,
                        "char_desc": desc,
                        "tokenizer": tok.name,
                        "clean_token_count": len(clean),
                        "injected_token_count": -1,
                        "delta": -99,
                        "survived": False,
                        "injected_string_repr": repr(injected),
                    })

    # Test 2: Homoglyph substitution
    for latin, cyrillic, desc in HOMOGLYPH_PAIRS:
        for verb in ["cancel", "start"]:
            if latin not in verb:
                continue
            substituted = verb.replace(latin, cyrillic)
            for tok in tokenizers:
                try:
                    ids_clean = tok.encode(verb)
                    ids_sub = tok.encode(substituted)
                    results.append({
                        "test": "homoglyph_substitution",
                        "verb": verb,
                        "char_code": f"{latin}->{cyrillic}",
                        "char_desc": desc,
                        "tokenizer": tok.name,
                        "clean_token_count": len(ids_clean),
                        "injected_token_count": len(ids_sub),
                        "delta": len(ids_sub) - len(ids_clean),
                        "survived": True,
                        "injected_string_repr": repr(substituted),
                    })
                except Exception as exc:
                    results.append({
                        "test": "homoglyph_substitution",
                        "verb": verb,
                        "char_code": f"{latin}->{cyrillic}",
                        "char_desc": desc,
                        "tokenizer": tok.name,
                        "clean_token_count": -1,
                        "injected_token_count": -1,
                        "delta": -99,
                        "survived": False,
                        "injected_string_repr": repr(substituted),
                    })

    return results


def _encode_all(text: str, tokenizers) -> dict[str, list[int]]:
    """Encode text with all tokenizers, return {name: ids}."""
    result = {}
    for tok in tokenizers:
        try:
            result[tok.name] = tok.encode(text)
        except Exception:
            result[tok.name] = []
    return result


def _char_survives(tok, char: str) -> bool:
    """Check if a character survives tokenization round-trip."""
    try:
        ids = tok.encode(char)
        decoded = tok.decode(ids)
        return char in decoded
    except Exception:
        return False


# ── 2. Trigger-head tokenization truncation ───────────────────────────

# SILP trigger headers to test
TRIGGER_HEADERS = [
    # Standard header
    "//silp:v1",
    "//silp:v2",
    # Alternative headers (spec section 3)
    "$silp:v1",
    "##silp:v1",
    "#silp:v1",
    # Short forms
    "//s:v1",
    "$s:v1",
    # Version variants (1-char version)
    "//silp:v0",
    "//silp:v9",
    # Bare markers (no protocol name)
    "//",
    "$",
    "##",
    # Header + first verb (realistic context)
    "//silp:v1 cancel(flight)",
    "$silp:v1 cancel(flight)",
    # Header with newline (multi-line payload)
    "//silp:v1\\ncancel(flight)",
]


def probe_trigger_head(tokenizers) -> list[dict[str, object]]:
    """Test how SILP trigger headers tokenize across tokenizer families.

    A good trigger header should:
    1. Be detectable as a distinct prefix (not merged with payload)
    2. Tokenize consistently across families
    3. Survive round-trip without mangled bytes
    """
    results: list[dict[str, object]] = []

    for header in TRIGGER_HEADERS:
        for tok in tokenizers:
            try:
                ids = tok.encode(header)
                tokens = [tok.decode([i]) for i in ids]
                results.append({
                    "header": header,
                    "tokenizer": tok.name,
                    "token_count": len(ids),
                    "tokens": json.dumps(tokens, ensure_ascii=False),
                    "first_token": tokens[0] if tokens else "",
                    "last_token": tokens[-1] if tokens else "",
                    "roundtrip_ok": tok.decode(ids) == header,
                })
            except Exception as exc:
                results.append({
                    "header": header,
                    "tokenizer": tok.name,
                    "token_count": -1,
                    "tokens": f"ERROR: {exc}",
                    "first_token": "",
                    "last_token": "",
                    "roundtrip_ok": False,
                })

    return results


# ── 3. Token-count variance threshold ─────────────────────────────────


def probe_variance_threshold(tokenizers, threshold: int = 3) -> list[dict[str, object]]:
    """Measure token-count variance for all frontend outputs.

    For each (case, frontend) pair, compute the max range across
    tokenizers. Flag any where range > threshold.
    """
    from silp.frontend import get_frontend, list_frontends
    from silp.ir import validate as validate_ir

    examples_dir = ROOT / "examples"
    results: list[dict[str, object]] = []

    for ir_path in sorted(examples_dir.glob("case*.json")):
        data = json.loads(ir_path.read_text(encoding="utf-8"))
        result = validate_ir(data)
        if not result.valid:
            continue

        case_id = ir_path.stem
        for fe_name in list_frontends():
            fe = get_frontend(fe_name)
            try:
                output = fe.compile(result.ir)
            except Exception:
                continue

            counts: list[int] = []
            tok_names: list[str] = []
            for tok in tokenizers:
                try:
                    ids = tok.encode(output)
                    counts.append(len(ids))
                    tok_names.append(tok.name)
                except Exception:
                    counts.append(-1)
                    tok_names.append(tok.name)

            valid_counts = [c for c in counts if c >= 0]
            if not valid_counts:
                continue

            min_c = min(valid_counts)
            max_c = max(valid_counts)
            range_c = max_c - min_c
            mean_c = sum(valid_counts) / len(valid_counts)

            results.append({
                "case_id": case_id,
                "frontend": fe_name,
                "char_count": len(output),
                "min_tokens": min_c,
                "max_tokens": max_c,
                "range": range_c,
                "mean": round(mean_c, 2),
                "exceeds_threshold": range_c > threshold,
                "threshold": threshold,
                "per_tokenizer": json.dumps(
                    dict(zip(tok_names, counts)), ensure_ascii=False
                ),
            })

    return results


# ── CSV writer ────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SILP Phase 2 probes — zerowidth/homoglyph + trigger-head + variance"
    )
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace tokenizers")
    parser.add_argument("--threshold", type=int, default=3, help="Variance threshold (default: 3)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building tokenizers...", file=sys.stderr)
    tokenizers = build_tokenizers(include_hf=not args.no_hf)
    if not tokenizers:
        print("No tokenizers available.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(tokenizers)} tokenizer(s): {[t.name for t in tokenizers]}", file=sys.stderr)

    # 1. Zero-width / homoglyph probe
    print("\n[1/3] Zero-width / homoglyph probe...", file=sys.stderr)
    zw_results = probe_zerowidth(tokenizers)
    zw_path = OUTPUT_DIR / "probe_zerowidth.csv"
    _write_csv(zw_path, zw_results)
    print(f"  -> {zw_path} ({len(zw_results)} rows)", file=sys.stderr)

    # Summary: how many injections changed token count?
    delta_count = sum(1 for r in zw_results if isinstance(r["delta"], int) and r["delta"] > 0)
    survived_count = sum(1 for r in zw_results if r["survived"])
    print(f"  Token count changed: {delta_count}/{len(zw_results)}", file=sys.stderr)
    print(f"  Char survived round-trip: {survived_count}/{len(zw_results)}", file=sys.stderr)

    # 2. Trigger-head truncation
    print("\n[2/3] Trigger-head tokenization probe...", file=sys.stderr)
    th_results = probe_trigger_head(tokenizers)
    th_path = OUTPUT_DIR / "probe_trigger_head.csv"
    _write_csv(th_path, th_results)
    print(f"  -> {th_path} ({len(th_results)} rows)", file=sys.stderr)

    # Summary: which headers have round-trip failures?
    rt_fail = [r for r in th_results if not r["roundtrip_ok"]]
    if rt_fail:
        print(f"  WARNING: {len(rt_fail)} round-trip failures:", file=sys.stderr)
        for r in rt_fail[:5]:
            print(f"    {r['header']} / {r['tokenizer']}: tokens={r['tokens']}",
                  file=sys.stderr)
    else:
        print(f"  All headers round-trip OK", file=sys.stderr)

    # 3. Variance threshold
    print(f"\n[3/3] Token-count variance (threshold={args.threshold})...", file=sys.stderr)
    var_results = probe_variance_threshold(tokenizers, args.threshold)
    var_path = OUTPUT_DIR / "probe_variance_threshold.csv"
    _write_csv(var_path, var_results)
    print(f"  -> {var_path} ({len(var_results)} rows)", file=sys.stderr)

    exceeded = [r for r in var_results if r["exceeds_threshold"]]
    if exceeded:
        print(f"  WARNING: {len(exceeded)} string(s) exceed threshold:", file=sys.stderr)
        for r in exceeded[:5]:
            print(f"    {r['case_id']}/{r['frontend']}: range={r['range']} "
                  f"(min={r['min_tokens']}, max={r['max_tokens']})", file=sys.stderr)
    else:
        print(f"  All strings within threshold", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
