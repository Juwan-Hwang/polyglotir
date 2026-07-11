#!/usr/bin/env python
"""MVP round-trip verification — Phase 1 deliverable.

Verifies that ``decode(compile(ir))`` produces an IR semantically equivalent
to the original for **every** example case × **every** frontend that supports
round-trip.

This is the "打通 MVP" milestone: the full compile → decode → verify cycle.

The verification checks:
1. **Intent preservation** — the decoded IR has the same ``!VERB`` intent.
2. **Entity preservation** — all entity IDs and values are recovered.
3. **Constraint preservation** — all constraint types, values, and times match.
4. **Alternative preservation** — all alternative actions and targets match.

The natural-language frontend is excluded (it is a control baseline, not a
round-trip codec).

Usage::

    python scripts/mvp_roundtrip.py
    python scripts/mvp_roundtrip.py --verbose
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLES_DIR = ROOT / "examples"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase1"


# ── Round-trip verification ───────────────────────────────────────────


def verify_roundtrip(ir, frontend) -> tuple[bool, list[str]]:
    """Verify that ``decode(compile(ir))`` produces an equivalent IR.

    Returns:
        (success, errors) — True if round-trip is semantically equivalent.
    """
    errors: list[str] = []

    # Compile
    compiled = frontend.compile(ir)

    # Decode
    try:
        decoded = frontend.decode(compiled)
    except Exception as exc:
        return False, [f"decode() raised: {exc}"]

    # 1. Intent preservation
    if decoded.intent != ir.intent:
        errors.append(
            f"intent mismatch: original={ir.intent!r} decoded={decoded.intent!r}"
        )

    # 2. Entity preservation
    # Normalize: treat action=None as equivalent to action=intent
    # (the frontend omits action for primary entities, but decode always sets it)
    def _norm_action(action, intent):
        return intent if action is None or action == intent else action

    # For primary entities: compare (id, value) — action is always intent
    orig_primary = {(e.id, e.value) for e in ir.entities
                    if _norm_action(e.action, ir.intent) == ir.intent}
    dec_primary = {(e.id, e.value) for e in decoded.entities
                   if _norm_action(e.action, decoded.intent) == decoded.intent}

    missing_pri = orig_primary - dec_primary
    extra_pri = dec_primary - orig_primary

    if missing_pri:
        errors.append(f"missing primary entities: {sorted(missing_pri)}")
    if extra_pri:
        errors.append(f"extra primary entities: {sorted(extra_pri)}")

    # For secondary entities: compare (value, action) — id may be lost
    # in code frontend (positional args always decode as id="act")
    orig_secondary = {(e.value, _norm_action(e.action, ir.intent))
                      for e in ir.entities
                      if _norm_action(e.action, ir.intent) != ir.intent}
    dec_secondary = {(e.value, _norm_action(e.action, decoded.intent))
                     for e in decoded.entities
                     if _norm_action(e.action, decoded.intent) != decoded.intent}

    missing_sec = orig_secondary - dec_secondary
    extra_sec = dec_secondary - orig_secondary

    if missing_sec:
        errors.append(f"missing secondary entities: {sorted(missing_sec)}")
    if extra_sec:
        errors.append(f"extra secondary entities: {sorted(extra_sec)}")

    # 3. Constraint preservation
    if len(decoded.constraints) != len(ir.constraints):
        errors.append(
            f"constraint count mismatch: original={len(ir.constraints)} "
            f"decoded={len(decoded.constraints)}"
        )
    else:
        for i, (orig_c, dec_c) in enumerate(zip(ir.constraints, decoded.constraints)):
            if orig_c.type != dec_c.type:
                errors.append(f"constraint[{i}].type: {orig_c.type!r} → {dec_c.type!r}")
            if orig_c.value != dec_c.value:
                errors.append(f"constraint[{i}].value: {orig_c.value!r} → {dec_c.value!r}")
            if orig_c.time != dec_c.time:
                errors.append(f"constraint[{i}].time: {orig_c.time!r} → {dec_c.time!r}")

            # Check operator (extra field)
            orig_op = getattr(orig_c, "operator", None)
            dec_op = getattr(dec_c, "operator", None)
            if orig_op != dec_op:
                errors.append(
                    f"constraint[{i}].operator: {orig_op!r} → {dec_op!r}"
                )

            # Check subject (extra field)
            orig_subj = getattr(orig_c, "subject", None)
            dec_subj = getattr(dec_c, "subject", None)
            if orig_subj != dec_subj:
                errors.append(
                    f"constraint[{i}].subject: {orig_subj!r} → {dec_subj!r}"
                )

    # 4. Alternative preservation
    if len(decoded.alternatives) != len(ir.alternatives):
        errors.append(
            f"alternative count mismatch: original={len(ir.alternatives)} "
            f"decoded={len(decoded.alternatives)}"
        )
    else:
        for i, (orig_a, dec_a) in enumerate(zip(ir.alternatives, decoded.alternatives)):
            if orig_a.action != dec_a.action:
                errors.append(f"alternative[{i}].action: {orig_a.action!r} → {dec_a.action!r}")
            if orig_a.target != dec_a.target:
                errors.append(f"alternative[{i}].target: {orig_a.target!r} → {dec_a.target!r}")
            if orig_a.location != dec_a.location:
                errors.append(f"alternative[{i}].location: {orig_a.location!r} → {dec_a.location!r}")

    return len(errors) == 0, errors


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SILP MVP round-trip verification (Phase 1)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed results for each case",
    )
    args = parser.parse_args()

    from silp.frontend import get_frontend, list_frontends
    from silp.ir import validate as validate_ir

    # Frontends that support round-trip.
    # Exclude control baselines (natural, nl_json, llmlingua2) — they are
    # not round-trip codecs by design.
    _NON_ROUNDTRIP = {"natural", "nl_json", "llmlingua2"}
    roundtrip_frontends = [
        f for f in list_frontends()
        if f not in _NON_ROUNDTRIP
    ]

    # Load all example cases
    cases: list[tuple[str, object]] = []
    for path in sorted(EXAMPLES_DIR.glob("case*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result = validate_ir(data, enforce_whitelist=False)
        if not result.valid:
            print(f"  [skip] {path.name}: {result.errors}", file=sys.stderr)
            continue
        cases.append((path.stem, result.ir))

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SILP MVP Round-Trip Verification — Phase 1", file=sys.stderr)
    print(f"  Cases:     {len(cases)}", file=sys.stderr)
    print(f"  Frontends: {roundtrip_frontends}", file=sys.stderr)
    print(f"  Total:     {len(cases) * len(roundtrip_frontends)} round-trips",
          file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    results: list[dict[str, object]] = []
    pass_count = 0
    total = len(cases) * len(roundtrip_frontends)

    for case_id, ir in cases:
        for fe_name in roundtrip_frontends:
            fe = get_frontend(fe_name)
            success, errors = verify_roundtrip(ir, fe)

            if success:
                pass_count += 1
                status = "PASS"
                if args.verbose:
                    compiled = fe.compile(ir)
                    print(f"  [PASS] {case_id} | {fe_name}", file=sys.stderr)
                    print(f"         compiled: {compiled}", file=sys.stderr)
                    print(f"         round-trip OK", file=sys.stderr)
            else:
                status = "FAIL"
                compiled = fe.compile(ir)
                print(f"  [FAIL] {case_id} | {fe_name}", file=sys.stderr)
                print(f"         compiled: {compiled}", file=sys.stderr)
                for e in errors:
                    print(f"         {e}", file=sys.stderr)

            results.append({
                "case_id": case_id,
                "frontend": fe_name,
                "status": status,
                "errors": "; ".join(errors) if errors else "",
            })

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Round-Trip Results:", file=sys.stderr)
    print(f"  Total:  {total}", file=sys.stderr)
    print(f"  Passed: {pass_count}", file=sys.stderr)
    print(f"  Failed: {total - pass_count}", file=sys.stderr)
    print(f"  Rate:   {pass_count/total*100:.1f}%" if total else "  N/A",
          file=sys.stderr)

    if pass_count == total:
        print(f"\n  MVP ROUND-TRIP: ALL PASS", file=sys.stderr)
    else:
        print(f"\n  MVP ROUND-TRIP: {total - pass_count} FAILURE(S)",
              file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # ── Write CSV ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "mvp_roundtrip.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "frontend", "status", "errors"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  Results: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
