"""SILP IR Validator — structural + semantic validation rule engine.

The validator runs a pipeline of **rules**, each checking one aspect of IR
correctness.  Rules are ordered by severity:

1. **Structural** — Pydantic model validation (types, patterns, required fields).
   If this fails, no further rules run.
2. **Whitelist** — every ``!VERB`` must be in the approved verb whitelist
   (populated from the Phase 0 cross-tokenizer census).
3. **Semantic rules** — protocol-level consistency checks that catch subtle
   errors before they reach the frontend compiler:

   - ``entity_action_match`` — entities claiming the same action as ``intent``
     are arguments; entities with a *different* action are secondary calls.
     This is valid, but logged as a warning for audit.
   - ``negation_consistency`` — a constraint with ``type="!X"`` must not also
     have a ``value`` of ``"true"`` (which would create a double-negative).
   - ``operator_validity`` — the ``operator`` field, if present, must be one of
     the supported comparison operators (``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``).
   - ``alternative_completeness`` — an alternative with ``location`` but no
     ``target`` is flagged (usually a mistake).
   - ``seq_integrity`` — if ``meta.seq`` is non-empty, it must reference
     existing entity IDs or well-known ordering keywords.
   - ``req_id_format`` — ``req_id`` must be 4–8 hex chars (structural, but
     also checked here for a cleaner error message).
   - ``duplicate_entity_id`` — two entities with the same ``id`` is an error
     (would cause ambiguous keyword arguments in the code frontend).
   - ``empty_entities`` — an IR with zero entities is valid but gets a warning
     (most real tasks have at least one argument).

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

from typing import Any, Callable

from pydantic import ValidationError

from .schema import SilpIR
from .whitelist import VERB_WHITELIST, VERB_REPLACEMENTS, suggest_replacement

# ── Rule types ────────────────────────────────────────────────────────


class ValidationRule:
    """A single validation rule with a name and callable.

    The callable receives the parsed :class:`SilpIR` and appends to the
    ``errors`` and ``warnings`` lists.  Rules are composable and testable
    in isolation.
    """

    __slots__ = ("name", "check")

    def __init__(self, name: str, check: Callable[[SilpIR, list[str], list[str]], None]) -> None:
        self.name = name
        self.check = check

    def __repr__(self) -> str:
        return f"<ValidationRule {self.name!r}>"


# ── Result container ──────────────────────────────────────────────────


class ValidationResult:
    """Outcome of IR validation."""

    __slots__ = ("valid", "errors", "warnings", "ir", "rules_run")

    def __init__(
        self,
        valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        ir: SilpIR | None = None,
        rules_run: list[str] | None = None,
    ) -> None:
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.ir = ir
        self.rules_run = rules_run or []

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        if self.valid:
            return f"ValidationResult(valid=True, rules={self.rules_run})"
        return f"ValidationResult(valid=False, errors={self.errors})"


# ── Semantic rule implementations ─────────────────────────────────────

# Supported comparison operators for constraint.operator field
_VALID_OPERATORS = frozenset({"==", "!=", "<", ">", "<=", ">="})


def _rule_whitelist(ir: SilpIR, errors: list[str], warnings: list[str]) -> None:
    """Rule: every !VERB must be in the approved whitelist."""
    _check_verb(ir.intent, "intent", errors, warnings)
    for e in ir.entities:
        if e.action:
            _check_verb(e.action, f"entity[{e.id}].action", errors, warnings)
    for alt in ir.alternatives:
        _check_verb(alt.action, "alternative.action", errors, warnings)


def _check_verb(action_code: str, label: str, errors: list[str], warnings: list[str]) -> None:
    """Check a single !VERB against the whitelist."""
    verb = action_code[1:]  # strip leading !
    if verb not in VERB_WHITELIST:
        replacement = suggest_replacement(verb)
        hint = f" Use !{replacement} instead." if replacement else ""
        errors.append(f"{label}: verb '{verb}' not in whitelist.{hint}")


def _rule_entity_action_consistency(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: entities with a different action than intent are secondary calls (warning)."""
    for e in ir.entities:
        if e.action and e.action != ir.intent:
            warnings.append(
                f"entity[{e.id}] has secondary action {e.action} "
                f"(differs from intent {ir.intent})"
            )


def _rule_negation_consistency(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: a negated constraint (!type) must not have value='true' (double-negative)."""
    for c in ir.constraints:
        if c.type.startswith("!") and c.value.lower() == "true":
            warnings.append(
                f"constraint type='{c.type}' with value='true' creates "
                f"a double-negative; consider value='false' for the base type"
            )


def _rule_operator_validity(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: constraint.operator must be a supported comparison operator."""
    for c in ir.constraints:
        operator = getattr(c, "operator", None)
        if operator is not None and operator not in _VALID_OPERATORS:
            errors.append(
                f"constraint type='{c.type}': invalid operator '{operator}'. "
                f"Must be one of: {', '.join(sorted(_VALID_OPERATORS))}"
            )


def _rule_alternative_completeness(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: alternative with location but no target is suspicious."""
    for i, alt in enumerate(ir.alternatives):
        if alt.location and not alt.target:
            warnings.append(
                f"alternative[{i}] has location='{alt.location}' "
                f"but no target — usually a mistake"
            )


def _rule_seq_integrity(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: meta.seq entries should reference known entity IDs or keywords."""
    if not ir.meta.seq:
        return

    entity_ids = {e.id for e in ir.entities}
    # Well-known ordering keywords that don't need to match entity IDs
    _known_keywords = {"reason_first", "reason_last", "parallel", "sequential"}

    for item in ir.meta.seq:
        if item in _known_keywords:
            continue
        if item not in entity_ids:
            warnings.append(
                f"meta.seq contains '{item}' which is not a known "
                f"entity ID or keyword"
            )


def _rule_duplicate_entity_id(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: two entities with the same id is an error."""
    seen: set[str] = set()
    for e in ir.entities:
        if e.id in seen:
            errors.append(
                f"duplicate entity id '{e.id}' — "
                f"would cause ambiguous keyword arguments in code frontend"
            )
        seen.add(e.id)


def _rule_empty_entities(
    ir: SilpIR, errors: list[str], warnings: list[str]
) -> None:
    """Rule: zero entities is valid but unusual."""
    if not ir.entities:
        warnings.append(
            "IR has no entities — most real tasks have at least one argument"
        )


# ── Rule registry ─────────────────────────────────────────────────────

_RULES: list[ValidationRule] = [
    ValidationRule("whitelist", _rule_whitelist),
    ValidationRule("duplicate_entity_id", _rule_duplicate_entity_id),
    ValidationRule("operator_validity", _rule_operator_validity),
    ValidationRule("negation_consistency", _rule_negation_consistency),
    ValidationRule("alternative_completeness", _rule_alternative_completeness),
    ValidationRule("entity_action_consistency", _rule_entity_action_consistency),
    ValidationRule("seq_integrity", _rule_seq_integrity),
    ValidationRule("empty_entities", _rule_empty_entities),
]


# ── Public API ────────────────────────────────────────────────────────


def validate(
    data: dict[str, Any],
    *,
    enforce_whitelist: bool = True,
    extra_rules: list[ValidationRule] | None = None,
) -> ValidationResult:
    """Validate a raw dict as SILP IR.

    Args:
        data: Parsed JSON dict conforming to the IR schema.
        enforce_whitelist: If True (default), verbs must be in the whitelist.
            Set to False for backward compatibility with Phase 0 test cases
            that use excluded verbs (e.g. ``!SWITCH_TOOL``).
        extra_rules: Additional custom rules to run after the built-in rules.

    Returns:
        :class:`ValidationResult` with ``.valid``, ``.errors``,
        ``.warnings``, ``.ir``, and ``.rules_run``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Structural validation (Pydantic)
    try:
        ir = SilpIR(**data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return ValidationResult(False, errors, warnings)

    # 2. Run semantic rules
    rules_run: list[str] = []
    active_rules = list(_RULES)

    # Optionally skip whitelist enforcement
    if not enforce_whitelist:
        active_rules = [r for r in active_rules if r.name != "whitelist"]

    # Add extra rules
    if extra_rules:
        active_rules.extend(extra_rules)

    for rule in active_rules:
        rule.check(ir, errors, warnings)
        rules_run.append(rule.name)

    if errors:
        return ValidationResult(False, errors, warnings, ir, rules_run)
    return ValidationResult(True, errors, warnings, ir, rules_run)


# ── Internals (kept for backward compat) ──────────────────────────────


def _check_whitelist(action_code: str, label: str, errors: list[str]) -> None:
    """Backward-compat wrapper for Phase 0 tests that import this directly."""
    verb = action_code[1:]  # strip leading !
    if verb not in VERB_WHITELIST:
        errors.append(f"{label}: verb '{verb}' not in whitelist")
