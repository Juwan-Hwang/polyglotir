"""Validate every JSON line in heartbeat results — detect truncated/partial writes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HB_DIR = ROOT / "data" / "raw" / "phase3_heartbeat"

all_ok = True
for jsonl in sorted(HB_DIR.glob("*/results.jsonl")):
    model = jsonl.parent.name
    lines = jsonl.read_bytes().split(b"\n")
    # Remove trailing empty line
    if lines and lines[-1] == b"":
        lines = lines[:-1]

    bad_lines = []
    for i, raw in enumerate(lines):
        try:
            obj = json.loads(raw)
            # Check required fields
            required = ["model", "n_value", "seed", "turn", "first_pass", "model_response"]
            missing = [f for f in required if f not in obj]
            if missing:
                bad_lines.append((i + 1, f"missing fields: {missing}"))
        except json.JSONDecodeError as e:
            bad_lines.append((i + 1, f"JSON decode error: {e}"))

    status = "OK" if not bad_lines else "CORRUPT"
    print(f"{model}: {len(lines)} lines, {len(bad_lines)} bad → {status}")
    if bad_lines:
        all_ok = False
        for lineno, err in bad_lines:
            print(f"  line {lineno}: {err}")
            # Show the raw bytes around the error
            raw = lines[lineno - 1]
            print(f"    raw (first 200 bytes): {raw[:200]}")
            print(f"    raw (last 200 bytes):  {raw[-200:]}")

# Also check backup integrity
print()
backup_dir = ROOT / "data" / "raw" / "phase3_heartbeat_backup_810"
if backup_dir.exists():
    for jsonl in sorted(backup_dir.glob("*/results.jsonl")):
        model = jsonl.parent.name
        lines = jsonl.read_bytes().split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]
        bad = 0
        for raw in lines:
            try:
                json.loads(raw)
            except json.JSONDecodeError:
                bad += 1
        print(f"backup/{model}: {len(lines)} lines, {bad} bad → {'OK' if bad == 0 else 'CORRUPT'}")
else:
    print("Backup directory not found!")

print()
print("OVERALL:", "ALL OK" if all_ok else "CORRUPTION DETECTED")
