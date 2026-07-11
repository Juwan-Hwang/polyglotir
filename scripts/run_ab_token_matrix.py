#!/usr/bin/env python
"""A/B test token-count matrix — Phase 1 missing experiment.

Runs the variant matrix (3 granularities x 3 containers = 9 variants)
for every example case x every frontend x every tokenizer, measuring
real token counts.

This is the experiment that was missing from Phase 1: the A/B framework
was built and unit-tested, but never run with real tokenizers to produce
actual token-count data.

Output:
    data/processed/phase1/ab_token_matrix.csv

Usage::

    python scripts/run_ab_token_matrix.py
    python scripts/run_ab_token_matrix.py --no-hf   # tiktoken only
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EXAMPLES_DIR = ROOT / "examples"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase1"

# Reuse tokenizer infrastructure from Phase 0 census
from tokenizer_census import build_tokenizers  # type: ignore[import-not-found]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="A/B test token-count matrix (Phase 1)")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace tokenizers")
    args = parser.parse_args()

    from silp.bench.ab_test import generate_variants, compile_variant
    from silp.frontend import list_frontends
    from silp.ir import validate as validate_ir

    # Build tokenizers
    print("Building tokenizers...", file=sys.stderr)
    tokenizers = build_tokenizers(include_hf=not args.no_hf)
    if not tokenizers:
        print("No tokenizers available.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(tokenizers)} tokenizer(s): {[t.name for t in tokenizers]}", file=sys.stderr)

    # Frontends to test (only round-trip frontends for variant compilation)
    frontend_names = ["code", "json", "natural"]

    # Load all example cases
    cases: list[tuple[str, object]] = []
    for path in sorted(EXAMPLES_DIR.glob("case*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result = validate_ir(data, enforce_whitelist=False)
        if not result.valid:
            print(f"  [skip] {path.name}: {result.errors}", file=sys.stderr)
            continue
        cases.append((path.stem, result.ir))

    print(f"\n  Cases:     {len(cases)}", file=sys.stderr)
    print(f"  Frontends: {frontend_names}", file=sys.stderr)
    print(f"  Variants:  9 (3 granularities x 3 containers)", file=sys.stderr)

    total_runs = len(cases) * 9 * len(frontend_names) * len(tokenizers)
    print(f"  Total:     {total_runs} token measurements\n", file=sys.stderr)

    # Run the matrix
    all_rows: list[dict[str, object]] = []

    for case_id, ir in cases:
        variants = generate_variants(ir)

        for variant in variants:
            for fe_name in frontend_names:
                try:
                    compiled = compile_variant(variant, fe_name)
                except Exception as exc:
                    print(f"  [error] {case_id}/{variant.label}/{fe_name}: {exc}", file=sys.stderr)
                    continue

                for tok in tokenizers:
                    try:
                        ids = tok.encode(compiled)
                        tc = len(ids)
                    except Exception:
                        tc = -1

                    all_rows.append({
                        "case_id": case_id,
                        "variant": variant.label,
                        "granularity": variant.granularity,
                        "container": variant.container,
                        "frontend": fe_name,
                        "tokenizer": tok.name,
                        "token_count": tc,
                        "char_count": len(compiled),
                        "compiled": compiled[:200],  # truncate for CSV
                    })

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "ab_token_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"A/B Token Matrix complete!", file=sys.stderr)
    print(f"  Total rows: {len(all_rows)}", file=sys.stderr)
    print(f"  Output:     {csv_path}", file=sys.stderr)

    # Quick summary: average token count per granularity x container
    from collections import defaultdict
    import statistics

    agg: dict[str, list[int]] = defaultdict(list)
    for r in all_rows:
        if r["token_count"] > 0:
            agg[r["variant"]].append(r["token_count"])

    print(f"\n  Summary (avg tokens across all cases/frontends/tokenizers):", file=sys.stderr)
    for variant in sorted(agg.keys()):
        vals = agg[variant]
        print(f"    {variant:20s}: mean={statistics.mean(vals):.1f}  "
              f"min={min(vals)}  max={max(vals)}", file=sys.stderr)

    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
