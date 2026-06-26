"""SILP IR Schema — Layer 1: Semantic Intermediate Representation.

Defines the coarse-grained task-slot IR as JSON Schema (Pydantic v2 models).
All action codes are IR primitives following the format: ``!`` + ``UPPERCASE_VERB``.

The IR is the canonical semantic representation. Frontends (Layer 2) compile
IR → surface string and decode surface string → IR. Every encoding produced
by any frontend MUST be losslessly decodable back to this IR.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Constants ──────────────────────────────────────────────────────────

SILP_VERSION = "v1"
IR_VERSION = "ir-v0.1"

# Action code pattern: ! followed by UPPERCASE verb (letters + underscores)
ACTION_CODE_RE = re.compile(r"^![A-Z][A-Z_]*$")

# req_id: 4–8 hex chars (start at 4, expand if collisions > 1% in Phase 1)
REQ_ID_RE = re.compile(r"^[a-f0-9]{4,8}$")


# ── Sub-models ────────────────────────────────────────────────────────


class Entity(BaseModel):
    """An entity slot in the task IR.

    If ``action`` is present and matches ``intent``, the entity is an argument
    to that action. If ``action`` differs from ``intent``, it is a secondary
    action (e.g. ``email(zhangsan)`` alongside ``cancel(flight)``).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    value: str
    action: Optional[str] = None  # !VERB

    @field_validator("action")
    @classmethod
    def _validate_action_code(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not ACTION_CODE_RE.match(v):
            raise ValueError(f"Invalid action code {v!r}: must be !UPPERCASE_VERB")
        return v


class Constraint(BaseModel):
    """A condition/constraint on the task.

    Convention for negation: prefix ``type`` with ``!`` (e.g. ``"!rain"``)
    to produce ``!rain(t+1)`` in the code frontend.

    Extra fields are allowed (e.g. ``subject``, ``operator``) and must be
    silently ignored by receivers that do not understand them.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    value: str
    time: Optional[str] = None  # e.g. "t+1", "t+1am"


class Alternative(BaseModel):
    """A fallback action (else-branch) when constraints are met."""

    model_config = ConfigDict(extra="allow")

    action: str  # !VERB
    target: Optional[str] = None
    location: Optional[str] = None

    @field_validator("action")
    @classmethod
    def _validate_action_code(cls, v: str) -> str:
        if not ACTION_CODE_RE.match(v):
            raise ValueError(f"Invalid action code {v!r}: must be !UPPERCASE_VERB")
        return v


class Meta(BaseModel):
    """Protocol-level metadata.

    Extension fields are allowed (``extra="allow"``) — receivers MUST silently
    ignore unknown fields, per the spec.
    """

    model_config = ConfigDict(extra="allow")

    priority: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    seq: list[str] = Field(default_factory=list)
    out: str = "natural"  # output format hint
    next_agent: Optional[str] = None
    req_id: str  # 4-digit short hash (expandable to 6–8)
    session_id: Optional[str] = None  # 2-char session ID for stateful mode

    @field_validator("req_id")
    @classmethod
    def _validate_req_id(cls, v: str) -> str:
        if not REQ_ID_RE.match(v):
            raise ValueError(f"Invalid req_id {v!r}: must be 4–8 hex chars")
        return v


# ── Root IR ───────────────────────────────────────────────────────────


class SilpIR(BaseModel):
    """Root SILP IR object — the semantic payload for MCP/A2A messages.

    Strict at the root level (``extra="forbid"``) — all extensibility lives
    inside ``meta`` or within sub-models that explicitly allow extras.
    """

    model_config = ConfigDict(extra="forbid")

    silp: str = Field(default=SILP_VERSION, pattern=r"^v\d+$")
    version: str = Field(default=IR_VERSION)
    intent: str  # main action: !VERB
    entities: list[Entity] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    meta: Meta

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, v: str) -> str:
        if not ACTION_CODE_RE.match(v):
            raise ValueError(f"Invalid intent {v!r}: must be !UPPERCASE_VERB")
        return v

    # ── Factory helpers ───────────────────────────────────────────

    @staticmethod
    def generate_req_id(content: str, length: int = 4) -> str:
        """Generate a short hash ``req_id`` from *content*.

        Phase 1 will simulate 1000 entries to test collision rate;
        if > 1%, expand to 6–8 chars.
        """
        return hashlib.sha256(content.encode()).hexdigest()[:length]

    def to_compact_json(self) -> str:
        """Serialize to compact JSON (no whitespace) — for hashing / transmission."""
        return self.model_dump_json(exclude_none=True)
