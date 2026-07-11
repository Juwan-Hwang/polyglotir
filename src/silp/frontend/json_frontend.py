"""JSON frontend — the pure-JSON-slot control baseline.

Renders the IR as a flat JSON object with only the semantic slots, no code
syntax.  Per spec §4 Phase 2, this baseline isolates "format contribution"
(structured JSON vs. code-like syntax) from "code vocabulary prior"
(function-call syntax that models have seen in training).

Phase 1: full round-trip — ``decode(compile(ir))`` reconstructs an
equivalent IR.
"""

from __future__ import annotations

import json

from ..ir.schema import Alternative, Constraint, Entity, SilpIR, Meta
from .base import Frontend


class JSONFrontend(Frontend):
    """JSON-slot frontend — flat semantic slots as JSON, no code syntax."""

    name = "json"

    # ── Compile: IR → str ─────────────────────────────────────────

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

    # ── Decode: str → IR ──────────────────────────────────────────

    def decode(self, text: str) -> SilpIR:
        """Decode JSON-frontend text back to IR.

        Reconstructs a :class:`SilpIR` from the flat JSON-slot format
        produced by :meth:`compile`.
        """
        data = json.loads(text)
        intent = data.pop("intent")

        entities: list[Entity] = []
        constraints: list[Constraint] = []
        alternatives: list[Alternative] = []

        # Pop non-entity slots first
        raw_constraints = data.pop("constraints", None)
        raw_alternatives = data.pop("alternatives", None)

        if raw_constraints:
            for c in raw_constraints:
                constraints.append(Constraint(**c))

        if raw_alternatives:
            for a in raw_alternatives:
                alternatives.append(Alternative(**a))

        # Remaining keys are entity slots
        # Keys starting with "action_" are secondary action groups
        for key, value in data.items():
            if key.startswith("action_"):
                # Secondary action: action_<verb> → list of values
                action = "!" + key[7:].upper()
                if isinstance(value, list):
                    for v in value:
                        entities.append(Entity(
                            id=key,  # use slot key as entity id
                            value=v,
                            action=action,
                        ))
                else:
                    entities.append(Entity(
                        id=key,
                        value=str(value),
                        action=action,
                    ))
            else:
                # Primary entity (action matches intent)
                entities.append(Entity(
                    id=key,
                    value=str(value),
                    action=intent,
                ))

        return SilpIR(
            intent=intent,
            entities=entities,
            constraints=constraints,
            alternatives=alternatives,
            meta=Meta(req_id=SilpIR.generate_req_id(text)),
        )
