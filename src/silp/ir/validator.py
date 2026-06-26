"""SILP IR Validator — structural + semantic validation.

The validator runs three checks:
1. **Structural** — Pydantic model validation (types, patterns, required fields).
2. **Whitelist** — if the verb whitelist is populated, every ``!VERB`` must be
   in the set (Phase 1 will populate the whitelist via the four-criteria filter).
3. **Consistency** — entity actions that differ from ``intent`` are flagged
   as secondary actions (valid, but logged for audit).

Usage::

    from silp.ir import validate
    result = validate(ir_dict)
    if result.valid:
        print(result.ir.to_compact_json())
    else:
        for e in result.errors:
            print(e)
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .schema import SilpIR

# ── Verb whitelist ────────────────────────────────────────────────────
#
# Empty set = allow any !UPPERCASE_VERB.
# Phase 1 will populate this via four criteria:
#   1. Single-token across all five tokenizers
#   2. Code-corpus frequency > 0.001 %
#   3. General-corpus frequency > 0.01 %
#   4. Strictly unambiguous within the protocol
# Phase 0 will also check "do sub-word fragments have other meanings?"
VERB_WHITELIST: set[str] = set()


# ── Result container ──────────────────────────────────────────────────


class ValidationResult:
    """Outcome of IR validation."""

    __slots__ = ("valid", "errors", "warnings", "ir")

    def __init__(
        self,
        valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        ir: SilpIR | None = None,
    ) -> None:
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.ir = ir

    def __bool__(self) -> bool:  # syntactic sugar: if result:
        return self.valid

    def __repr__(self) -> str:
        if self.valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={self.errors})"


# ── Public API ────────────────────────────────────────────────────────


def validate(data: dict[str, Any]) -> ValidationResult:
    """Validate a raw dict as SILP IR.

    Args:
        data: Parsed JSON dict conforming to the IR schema.

    Returns:
        :class:`ValidationResult` with ``.valid``, ``.errors``,
        ``.warnings``, and ``.ir`` (the parsed object, if valid).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Structural validation
    try:
        ir = SilpIR(**data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return ValidationResult(False, errors, warnings)

    # 2. Verb whitelist (only when populated — Phase 1+)
    if VERB_WHITELIST:
        _check_whitelist(ir.intent, "intent", errors)
        for e in ir.entities:
            if e.action:
                _check_whitelist(e.action, f"entity[{e.id}].action", errors)
        for alt in ir.alternatives:
            _check_whitelist(alt.action, "alternative.action", errors)

    # 3. Consistency audit (warnings, not errors)
    for e in ir.entities:
        if e.action and e.action != ir.intent:
            warnings.append(
                f"entity[{e.id}] has secondary action {e.action} "
                f"(differs from intent {ir.intent})"
            )

    if errors:
        return ValidationResult(False, errors, warnings, ir)
    return ValidationResult(True, errors, warnings, ir)


# ── Internals ─────────────────────────────────────────────────────────


def _check_whitelist(action_code: str, label: str, errors: list[str]) -> None:
    """Append an error if *action_code*'s verb is not in the whitelist."""
    verb = action_code[1:]  # strip leading !
    if verb not in VERB_WHITELIST:
        errors.append(f"{label}: verb '{verb}' not in whitelist")
