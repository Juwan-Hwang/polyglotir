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

Phase 1: full round-trip — ``decode(compile(ir))`` reconstructs an
equivalent IR.
"""

from __future__ import annotations

import re

from ..ir.schema import Alternative, Constraint, Entity, SilpIR, Meta
from .base import Frontend

# ── Helpers ───────────────────────────────────────────────────────────


def _verb_to_fn(action_code: str) -> str:
    """``!CANCEL`` → ``cancel``."""
    return action_code[1:].lower()


def _fn_to_verb(fn_name: str) -> str:
    """``cancel`` → ``!CANCEL``."""
    return "!" + fn_name.upper()


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

    # Regex patterns for parsing compiled code-frontend strings.

    # Top-level split: [if COND: ]ACTIONS[ else: ELSE]
    _RE_IF = re.compile(r"^if\s+(.+?):\s+(.+)$", re.DOTALL)
    _RE_ELSE = re.compile(r"\s+else:\s+(.+)$")

    # Function call: name(arg1,arg2,...)
    _RE_CALL = re.compile(r"^(\w+)\((.*)\)$", re.DOTALL)

    # Constraint patterns
    # Negation: !word(time) or !word
    _RE_NEGATION = re.compile(r"^!(\w+)(?:\(([^)]+)\))?$")
    # Operator: type OP value [, time]
    _RE_OPERATOR = re.compile(r"^(\w+)([<>=!]+)([\w.]+)(?:,\s*(\S+))?$")
    # Function call constraint: type(args...)
    _RE_FUNC_CONSTRAINT = re.compile(r"^(\w+)\(([^)]*)\)$")

    # Keyword arg: key=value
    _RE_KWARG = re.compile(r"^(\w+)=(.+)$")

    # Alternative target with location: target@location
    _RE_TARGET_LOC = re.compile(r"^([^@]+)@(.+)$")

    def decode(self, text: str) -> SilpIR:
        """Decode code-frontend text back to IR.

        Parses the mini-DSL produced by :meth:`compile` and reconstructs
        an equivalent :class:`SilpIR`.

        Raises:
            ValueError: if *text* cannot be parsed.
        """
        text = text.strip()

        # 1. Extract else-branch (if present)
        else_text = ""
        else_match = self._RE_ELSE.search(text)
        if else_match:
            else_text = else_match.group(1)
            text = text[: else_match.start()].strip()

        # 2. Extract if-condition (if present)
        constraints: list[Constraint] = []
        if_match = self._RE_IF.match(text)
        if if_match:
            cond_str = if_match.group(1).strip()
            action_str = if_match.group(2).strip()
            constraints = self._parse_constraints(cond_str)
        else:
            action_str = text

        # 3. Parse actions
        entities, intent = self._parse_actions(action_str)

        # 4. Parse else-branch
        alternatives = self._parse_alternatives(else_text) if else_text else []

        # 5. Reconstruct IR
        return SilpIR(
            intent=intent,
            entities=entities,
            constraints=constraints,
            alternatives=alternatives,
            meta=Meta(req_id=SilpIR.generate_req_id(text)),
        )

    def _parse_constraints(self, cond_str: str) -> list[Constraint]:
        """Parse a condition string into Constraint objects.

        Conditions are joined by `` and ``.
        Each condition is one of:
        - Negation: ``!rain(t+1)`` or ``!rain``
        - Operator: ``weather>rain`` or ``weather>rain, t+1``
        - Function: ``loc(me,Beijing,t+1am)``
        """
        constraints: list[Constraint] = []
        parts = _split_top_level(cond_str, " and ")

        for part in parts:
            part = part.strip()
            if not part:
                continue
            constraint = self._parse_single_constraint(part)
            if constraint:
                constraints.append(constraint)

        return constraints

    def _parse_single_constraint(self, part: str) -> Constraint | None:
        """Parse a single constraint expression."""
        # Try negation: !word(time) or !word
        m = self._RE_NEGATION.match(part)
        if m:
            neg_type = "!" + m.group(1)
            time = m.group(2) if m.group(2) else None
            return Constraint(type=neg_type, value="true", time=time)

        # Try operator: type OP value [, time]
        m = self._RE_OPERATOR.match(part)
        if m:
            ctype = m.group(1)
            operator = m.group(2)
            value = m.group(3)
            time = m.group(4) if m.group(4) else None
            c = Constraint(type=ctype, value=value, time=time)
            # Set operator as extra field
            object.__setattr__(c, "operator", operator)
            return c

        # Try function call: type(args...)
        m = self._RE_FUNC_CONSTRAINT.match(part)
        if m:
            ctype = m.group(1)
            args_str = m.group(2)
            args = [a.strip() for a in args_str.split(",") if a.strip()]

            # Determine subject, value, time from positional args
            subject = None
            value = ""
            time = None

            # Heuristic: if 3 args, [subject, value, time]
            # If 2 args, [value, time] or [subject, value]
            # If 1 arg, [value]
            if len(args) >= 3:
                subject = args[0]
                value = args[1]
                time = args[2]
            elif len(args) == 2:
                # Check if second arg looks like a time (t+...)
                if args[1].startswith("t+") or args[1].startswith("t-"):
                    value = args[0]
                    time = args[1]
                else:
                    # [subject, value]
                    subject = args[0]
                    value = args[1]
            elif len(args) == 1:
                value = args[0]

            c = Constraint(type=ctype, value=value, time=time)
            if subject:
                object.__setattr__(c, "subject", subject)
            return c

        # Unknown format
        raise ValueError(f"Cannot parse constraint: {part!r}")

    def _parse_actions(self, action_str: str) -> tuple[list[Entity], str]:
        """Parse action string into entities and intent.

        Returns:
            (entities, intent) where intent is the ``!VERB`` of the first call.
        """
        calls = _split_top_level(action_str, "; ")
        if not calls:
            raise ValueError(f"Empty action string: {action_str!r}")

        entities: list[Entity] = []
        intent = ""

        for i, call_str in enumerate(calls):
            call_str = call_str.strip()
            m = self._RE_CALL.match(call_str)
            if not m:
                raise ValueError(f"Cannot parse function call: {call_str!r}")

            fn_name = m.group(1)
            args_str = m.group(2)
            verb = _fn_to_verb(fn_name)

            if i == 0:
                intent = verb

            action = verb
            args = _split_top_level(args_str, ",")

            for arg in args:
                arg = arg.strip()
                if not arg:
                    continue
                kw_match = self._RE_KWARG.match(arg)
                if kw_match:
                    # Keyword arg: key=value
                    entity_id = kw_match.group(1)
                    value = kw_match.group(2)
                else:
                    # Positional arg → id="act"
                    entity_id = "act"
                    value = arg

                entities.append(Entity(
                    id=entity_id,
                    value=value,
                    action=action,
                ))

        if not intent:
            raise ValueError(f"Could not determine intent from: {action_str!r}")

        return entities, intent

    def _parse_alternatives(self, else_str: str) -> list[Alternative]:
        """Parse else-branch string into Alternative objects."""
        alternatives: list[Alternative] = []
        calls = _split_top_level(else_str, "; ")

        for call_str in calls:
            call_str = call_str.strip()
            m = self._RE_CALL.match(call_str)
            if not m:
                raise ValueError(f"Cannot parse alternative: {call_str!r}")

            fn_name = m.group(1)
            args_str = m.group(2)
            verb = _fn_to_verb(fn_name)

            args = [a.strip() for a in args_str.split(",") if a.strip()]

            target = None
            location = None

            for arg in args:
                # Check for target@location
                tl_match = self._RE_TARGET_LOC.match(arg)
                if tl_match:
                    target = tl_match.group(1)
                    location = tl_match.group(2)
                elif arg.startswith("loc="):
                    location = arg[4:]
                else:
                    target = arg

            alternatives.append(Alternative(
                action=verb,
                target=target,
                location=location,
            ))

        return alternatives


# ── Utility: top-level split that respects parentheses ───────────────


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* by *sep* at the top level (not inside parentheses).

    >>> _split_top_level("a,b(c,d),e", ",")
    ['a', 'b(c,d)', 'e']
    >>> _split_top_level("f1(a); f2(b)", "; ")
    ['f1(a)', 'f2(b)']
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0

    while i < len(text):
        char = text[i]

        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif depth == 0 and text[i:i + len(sep)] == sep:
            parts.append("".join(current))
            current = []
            i += len(sep)
            continue
        else:
            current.append(char)

        i += 1

    if current:
        parts.append("".join(current))

    return parts
