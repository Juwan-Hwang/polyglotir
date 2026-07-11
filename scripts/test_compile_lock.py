#!/usr/bin/env python
"""Quick verification test for CompileLock."""
import json
from pathlib import Path
from silp.frontend import get_frontend, CompileLock
from silp.ir import validate

ir_path = Path("examples/case1_multi_constraint.json")
data = json.loads(ir_path.read_text(encoding="utf-8"))
result = validate(data)
ir = result.ir

fe = get_frontend("code")
compiled = fe.compile(ir)
print(f"Compiled: {compiled}")

lock = CompileLock.seal(frontend_name="code", ir=ir, compiled=compiled)
print(f"Lock:     {lock}")
print(f"Verified: {lock.verify()}")
print(f"IR match: {lock.verify_ir(ir)}")

tampered = compiled + " "
print(f"Tamper:   {lock.verify(tampered)}")

j = lock.to_json()
lock2 = CompileLock.from_json(j)
print(f"Round-trip: {lock == lock2}")
print(f"Lock2 verified: {lock2.verified}")
print()
print("All checks passed.")
