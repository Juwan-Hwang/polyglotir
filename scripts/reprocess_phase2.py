#!/usr/bin/env python
"""Reprocess Phase 2 results from existing raw JSONL files.

Reads all per-model ``results.jsonl`` files and regenerates the
processed CSV/JSON outputs (matrix, case details, retries, Spearman).

This is the safe way to merge results from multiple runs without
risking overwriting valid data.

Usage::

    python scripts/reprocess_phase2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw" / "phase2"

# Import processing functions from the main script
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_phase2_matrix", ROOT / "scripts" / "run_phase2_matrix.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_write_matrix_csv = _mod._write_matrix_csv
_write_case_details = _mod._write_case_details
_write_retry_audit = _mod._write_retry_audit
_compute_spearman = _mod._compute_spearman
_write_spearman = _mod._write_spearman
from silp.frontend import list_frontends  # noqa: E402


def main() -> None:
    # Auto-discover models from subdirectories
    model_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    models: list[str] = []
    for d in model_dirs:
        f = d / "results.jsonl"
        if f.exists() and f.stat().st_size > 0:
            first = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
            models.append(first["model"])

    if not models:
        print("Error: no results.jsonl files found", file=sys.stderr)
        sys.exit(1)

    frontends = list_frontends()

    # Load all results
    all_results: list[dict] = []
    for m in models:
        slug = m.replace(".", "-").replace("/", "-")
        jsonl_path = RAW_DIR / slug / "results.jsonl"
        count = 0
        errors = 0
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            all_results.append(r)
            count += 1
            if r.get("judge") == "error":
                errors += 1
        print(f"  {m}: {count} runs ({errors} errors)", file=sys.stderr)

    print(f"\n  Total: {len(all_results)} runs, {len(models)} models, {len(frontends)} frontends", file=sys.stderr)

    # Regenerate all processed outputs
    _write_matrix_csv(all_results, frontends, models)
    _write_case_details(all_results)
    _write_retry_audit(all_results)
    spearman = _compute_spearman(all_results, frontends, models)
    _write_spearman(spearman)

    print(f"\n  Matrix:     {ROOT}/data/processed/phase2/phase2_matrix.csv", file=sys.stderr)
    print(f"  Spearman:   {ROOT}/data/processed/phase2/phase2_spearman.json", file=sys.stderr)
    print(f"  Case details: {ROOT}/data/processed/phase2/phase2_case_details.csv", file=sys.stderr)
    print(f"  Retries:    {ROOT}/data/processed/phase2/phase2_retries.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
