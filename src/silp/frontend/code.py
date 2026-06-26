"""Code frontend — the default frontend using function-call syntax.

Maps IR action codes (``!CANCEL``) to lowercase function calls (``cancel``)
and compiles constraints/entities/alternatives into a readable mini-DSL:

    if loc(me,Beijing,t+1am): cancel(flight,t+1pm); email(zhangsan)
    if !rain(t+1): start(hike) else start(cards@indoor)
    translate(src=fr_rev_bold, tgt=shakespeare_en, style=archaic_heavy)

Design rules (from spec §2 Layer 2):
- Code frontend → ``cancel()``; structure-symbol frontend → ``!CANCEL``.
  The two are NEVER mixed.
- Constraints with ``!``-prefixed ``type`` become negated conditions.
- Entities whose ``action`` matches ``intent`` are arguments; entities with
  a different ``action`` are secondary calls joined by ``;``.
- Alternatives compile to ``else`` branches.
"""

from __future__ import annotations

from ..ir.schema import Alternative, Constraint, Entity, SilpIR
from .base import Frontend

# ── Helpers ───────────────────────────────────────────────────────────


def _verb_to_fn(action_code: str) -> str:
    """``!CANCEL`` → ``cancel``."""
    return action_code[1:].lower()


def _entity_to_arg(entity: Entity) -> str:
    """Render an entity as a function-call argument.

    ``id == "act"`` → positional arg (bare value).
    Otherwise → keyword arg ``id=value``.
    """
    if entity.id == "act":
        return entity.value
    return f"{entity.id}={entity.value}"


# ── Frontend ──────────────────────────────────────────────────────────


class CodeFrontend(Frontend):
    """Code/type frontend — compiles IR to function-call-like syntax."""

    name = "code"

    # ── Compile: IR → str ─────────────────────────────────────────

    def compile(self, ir: SilpIR) -> str:
        condition = self._compile_constraints(ir.constraints)
        actions = self._compile_actions(ir.intent, ir.entities)
        else_branch = self._compile_alternatives(ir.alternatives)

        # Assemble: [if COND: ]ACTIONS[ else ELSE_BRANCH]
        parts: list[str] = []

        if condition:
            parts.append(f"if {condition}: {actions}")
        else:
            parts.append(actions)

        if else_branch:
            parts.append(f"else: {else_branch}")

        return " ".join(parts)

    def _compile_constraints(self, constraints: list[Constraint]) -> str:
        """Render constraints as a boolean condition string.

        - ``!``-prefixed type → negation: ``!rain(t+1)``
        - extra ``subject`` field → first positional arg: ``loc(me,Beijing,t+1am)``
        - extra ``operator`` field → infix: ``weather>rain``
        """
        if not constraints:
            return ""

        rendered: list[str] = []
        for c in constraints:
            # Collect positional args: [subject], value, [time]
            args: list[str] = []
            subject = getattr(c, "subject", None)
            if subject:
                args.append(subject)

            operator = getattr(c, "operator", None)
            if operator:
                # Infix form: type>value  (e.g. weather>rain)
                time_suffix = f", {c.time}" if c.time else ""
                rendered.append(f"{c.type}{operator}{c.value}{time_suffix}")
                continue

            if c.type.startswith("!"):
                # Negation: !rain(t+1)
                time_suffix = f"({c.time})" if c.time else ""
                rendered.append(f"{c.type}{time_suffix}")
            else:
                args.append(c.value)
                if c.time:
                    args.append(c.time)
                rendered.append(f"{c.type}({','.join(args)})")

        return " and ".join(rendered) if len(rendered) > 1 else rendered[0]

    def _compile_actions(self, intent: str, entities: list[Entity]) -> str:
        """Render the main action + any secondary actions.

        Entities matching ``intent`` → arguments to the primary call
        (``id="act"`` positional, others keyword).
        Entities with a different ``action`` → separate calls whose arguments
        are positional values, grouped by action and joined by ``;``.
        """
        primary_args: list[str] = []
        # action_code → list of positional values
        secondary_groups: dict[str, list[str]] = {}

        for e in entities:
            if e.action is None or e.action == intent:
                primary_args.append(_entity_to_arg(e))
            else:
                secondary_groups.setdefault(e.action, []).append(e.value)

        primary = f"{_verb_to_fn(intent)}({','.join(primary_args)})"
        all_calls = [primary]
        for action, values in secondary_groups.items():
            fn = _verb_to_fn(action)
            all_calls.append(f"{fn}({','.join(values)})")

        return "; ".join(all_calls)

    def _compile_alternatives(self, alternatives: list[Alternative]) -> str:
        """Render alternatives as ``else``-branch calls."""
        if not alternatives:
            return ""

        calls: list[str] = []
        for alt in alternatives:
            fn = _verb_to_fn(alt.action)
            args: list[str] = []
            if alt.target:
                loc_suffix = f"@{alt.location}" if alt.location else ""
                args.append(f"{alt.target}{loc_suffix}")
            elif alt.location:
                args.append(f"loc={alt.location}")
            calls.append(f"{fn}({','.join(args)})")

        return "; ".join(calls)

    # ── Decode: str → IR (Phase 1) ────────────────────────────────

    def decode(self, text: str) -> SilpIR:
        """Decode code-frontend text back to IR.

        .. note::
            Full round-trip decode is a Phase 1 deliverable.
            Phase 0.5 uses ``compile`` only (one-way) for smoke tests.
        """
        raise NotImplementedError(
            "CodeFrontend.decode is a Phase 1 deliverable. "
            "Phase 0.5 uses compile-only (one-way) for smoke tests."
        )
