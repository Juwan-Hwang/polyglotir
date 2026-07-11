"""Natural-language frontend — the control baseline.

Produces a plain-natural-language rendering of the IR.  This is NOT a
structured-natural-language variant (which could be accidentally parsed as
code); it is deliberately unstructured prose to cleanly isolate the
"format contribution" from the "code vocabulary prior".

Per spec §4 Phase 2: the natural-language control replaces the earlier
"structured natural language" baseline because the latter is too easily
parsed as code by modern models.
"""

from __future__ import annotations

from ..ir.schema import SilpIR
from .base import Frontend


class NaturalFrontend(Frontend):
    """Natural-language frontend — unstructured prose control baseline."""

    name = "natural"

    def compile(self, ir: SilpIR) -> str:
        clauses: list[str] = []

        # Condition clause
        if ir.constraints:
            cond_parts = []
            for c in ir.constraints:
                if c.type.startswith("!"):
                    negated = c.type[1:]
                    cond_parts.append(f"not {negated}" if not c.time
                                     else f"not {negated} at {c.time}")
                else:
                    operator = getattr(c, "operator", None)
                    if operator:
                        # Operator form: "status is not shipped", "budget is at most 500"
                        op_phrase = _OPERATOR_PHRASES.get(operator, operator)
                        fragment = f"{c.type} {op_phrase} {c.value}"
                    else:
                        fragment = f"{c.type} is {c.value}"
                    if c.time:
                        fragment += f" at {c.time}"
                    cond_parts.append(fragment)
            clauses.append("If " + " and ".join(cond_parts) + ",")

        # Main action clause
        main_verb = _verb_to_phrase(ir.intent)
        targets = [e.value for e in ir.entities
                   if e.action is None or e.action == ir.intent]
        obj = ", ".join(targets) if targets else "the task"
        clauses.append(f"{main_verb} {obj}")

        # Secondary actions
        for e in ir.entities:
            if e.action and e.action != ir.intent:
                verb = _verb_to_phrase(e.action)
                clauses.append(f"and {verb} {e.value}")

        # Alternative clause
        if ir.alternatives:
            alt_parts = []
            for alt in ir.alternatives:
                verb = _verb_to_phrase(alt.action)
                target = alt.target or "the alternative"
                if alt.location:
                    alt_parts.append(f"{verb} {target} indoors"
                                     if alt.location == "indoor"
                                     else f"{verb} {target} at {alt.location}")
                else:
                    alt_parts.append(f"{verb} {target}")
            clauses.append("otherwise " + " and ".join(alt_parts))

        return " ".join(clauses) + "."

    def decode(self, text: str) -> SilpIR:
        """Natural-language decode is inherently lossy — not supported.

        The natural frontend is a *control baseline*, not a round-trip
        codec.  It exists to measure how well models understand unstructured
        prose vs. structured code frontends.
        """
        raise NotImplementedError(
            "NaturalFrontend.decode is not supported — it is a control "
            "baseline, not a round-trip codec."
        )


# ── Helpers ───────────────────────────────────────────────────────────

# Minimal verb → natural-language phrase mapping.
# Phase 1 will expand this from the verb whitelist.
_VERB_PHRASES: dict[str, str] = {
    "CANCEL": "cancel",
    "START": "start",
    "EMAIL": "notify",
    "FETCH": "fetch",
    "PROCESS": "process",
    "TRANSLATE": "translate",
    "SWITCH_TOOL": "switch to",
    "BOOK": "book",
    "ROUTE": "route",
    "SEARCH": "search",
    "UPDATE": "update",
    "ESCALATE": "escalate",
    "SUGGEST": "suggest",
}


def _verb_to_phrase(action_code: str) -> str:
    """``!CANCEL`` → ``"cancel"`` (or a human-friendly phrase)."""
    verb = action_code[1:]
    return _VERB_PHRASES.get(verb, verb.lower())


# Operator → natural-language phrase mapping.
_OPERATOR_PHRASES: dict[str, str] = {
    "!=": "is not",
    "==": "is",
    "<=": "is at most",
    ">=": "is at least",
    "<": "is less than",
    ">": "is greater than",
}
