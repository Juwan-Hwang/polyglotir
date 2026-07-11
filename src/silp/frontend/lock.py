"""Compile lock — cryptographic immutability for compiled frontend outputs.

When a frontend compiles IR → ``str``, a :class:`CompileLock` binds the output
to its source IR via a SHA-256 hash pair.  This provides:

1. **Tamper detection** — any modification to the compiled string invalidates
   the lock.
2. **Provenance tracking** — the IR that produced the output is uniquely
   identified by ``ir_hash``.
3. **Audit trail** — every compilation is timestamped and serialisable to
   JSON for logging / transmission.

Lock record format (JSON)::

    {
        "frontend":   "code",
        "compiled":   "if loc(me,Beijing,t+1am): cancel(flight); email(zhangsan)",
        "ir_hash":    "sha256:a1b2c3d4…",
        "output_hash":"sha256:e5f6g7h8…",
        "timestamp":  "2026-07-11T02:00:00Z",
        "verified":   true
    }

Usage::

    from silp.frontend.lock import CompileLock

    lock = CompileLock.seal(frontend_name="code", ir=ir, compiled=output)
    assert lock.verified                     # output matches hash
    assert lock.verify(output)               # re-verify later
    lock_dict = lock.to_dict()               # serialise for logging
    lock2 = CompileLock.from_dict(lock_dict) # deserialise
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from ..ir.schema import SilpIR

# ── Constants ─────────────────────────────────────────────────────────

_HASH_PREFIX = "sha256:"
_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


# ── Helpers ───────────────────────────────────────────────────────────


def _sha256(data: str) -> str:
    """Return ``sha256:<hex>`` for *data*."""
    return _HASH_PREFIX + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)


# ── CompileLock ───────────────────────────────────────────────────────


class CompileLock:
    """Immutable record binding a compiled output to its source IR.

    Attributes:
        frontend:    Name of the frontend that produced the output.
        compiled:    The compiled surface string.
        ir_hash:     SHA-256 hash of the IR's compact JSON.
        output_hash: SHA-256 hash of the compiled string.
        timestamp:   UTC ISO-8601 string when the lock was sealed.
        verified:    Whether ``output_hash`` matches ``compiled``.
    """

    __slots__ = (
        "frontend",
        "compiled",
        "ir_hash",
        "output_hash",
        "timestamp",
        "verified",
    )

    def __init__(
        self,
        frontend: str,
        compiled: str,
        ir_hash: str,
        output_hash: str,
        timestamp: str,
        verified: bool,
    ) -> None:
        self.frontend = frontend
        self.compiled = compiled
        self.ir_hash = ir_hash
        self.output_hash = output_hash
        self.timestamp = timestamp
        self.verified = verified

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def seal(
        cls,
        frontend_name: str,
        ir: SilpIR,
        compiled: str,
        *,
        timestamp: str | None = None,
    ) -> CompileLock:
        """Create a lock binding *compiled* to *ir*.

        Args:
            frontend_name: Name of the frontend (e.g. ``"code"``).
            ir:            The source :class:`SilpIR`.
            compiled:      The compiled surface string.
            timestamp:     Override timestamp (UTC ISO-8601). Defaults to now.
        """
        ir_hash = _sha256(ir.to_compact_json())
        output_hash = _sha256(compiled)
        ts = timestamp or _now_iso()
        return cls(
            frontend=frontend_name,
            compiled=compiled,
            ir_hash=ir_hash,
            output_hash=output_hash,
            timestamp=ts,
            verified=True,
        )

    # ── Verification ──────────────────────────────────────────────

    def verify(self, compiled: str | None = None) -> bool:
        """Verify that *compiled* matches the locked ``output_hash``.

        Args:
            compiled: The string to check.  If ``None``, re-checks the
                stored ``self.compiled``.
        """
        text = compiled if compiled is not None else self.compiled
        return _sha256(text) == self.output_hash

    def verify_ir(self, ir: SilpIR) -> bool:
        """Verify that *ir* matches the locked ``ir_hash``."""
        return _sha256(ir.to_compact_json()) == self.ir_hash

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON logging / transmission."""
        return {
            "frontend": self.frontend,
            "compiled": self.compiled,
            "ir_hash": self.ir_hash,
            "output_hash": self.output_hash,
            "timestamp": self.timestamp,
            "verified": self.verify(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompileLock:
        """Deserialise from a dict (e.g. loaded from JSONL log)."""
        lock = cls(
            frontend=d["frontend"],
            compiled=d["compiled"],
            ir_hash=d["ir_hash"],
            output_hash=d["output_hash"],
            timestamp=d["timestamp"],
            verified=False,
        )
        lock.verified = lock.verify()
        return lock

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> CompileLock:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(s))

    # ── Dunder ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "OK" if self.verified else "FAIL"
        return (
            f"<CompileLock {status} frontend={self.frontend!r} "
            f"ir_hash={self.ir_hash[:16]}… "
            f"output_hash={self.output_hash[:16]}…>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompileLock):
            return NotImplemented
        return (
            self.frontend == other.frontend
            and self.ir_hash == other.ir_hash
            and self.output_hash == other.output_hash
        )
