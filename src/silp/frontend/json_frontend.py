"""JSON frontend — the pure-JSON-slot control baseline.

Renders the IR as a flat JSON object with only the semantic slots, no code
syntax.  Per spec §4 Phase 2, this baseline isolates "format contribution"
(structured JSON vs. code-like syntax) from "code vocabulary prior"
(function-call syntax that models have seen in training).
"""

from __future__ import annotations

import json

from ..ir.schema import SilpIR
from .base import Frontend


class JSONFrontend(Frontend):
    """JSON-slot frontend — flat semantic slots as JSON, no code syntax."""

    name = "json"

    def compile(self, ir: SilpIR) -> str:
        slots: dict[str, object] = {
            "intent": ir.intent,
        }

        # Flatten entities into top-level slots
        for e in ir.entities:
            if e.action is None or e.action == ir.intent:
                slots[e.id] = e.value
            else:
                # Secondary actions get their own slot
                key = f"action_{e.action[1:].lower()}"
                slots.setdefault(key, []).append(e.value)

        if ir.constraints:
            slots["constraints"] = [
                c.model_dump(exclude_none=True) for c in ir.constraints
            ]

        if ir.alternatives:
            slots["alternatives"] = [
                a.model_dump(exclude_none=True) for a in ir.alternatives
            ]

        return json.dumps(slots, ensure_ascii=False, separators=(",", ":"))

    def decode(self, text: str) -> SilpIR:
        """JSON frontend round-trip is technically possible but not a Phase 0.5 priority."""
        raise NotImplementedError(
            "JSONFrontend.decode is a Phase 1 deliverable."
        )
