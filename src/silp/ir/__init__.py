"""silp-ir — Layer 1: Semantic Intermediate Representation.

Provides the coarse-grained task-slot IR (JSON Schema via Pydantic v2),
a built-in validator with rule engine, and the verb whitelist machinery.
"""

from .schema import (
    SilpIR,
    Entity,
    Constraint,
    Alternative,
    Meta,
    ACTION_CODE_RE,
    SILP_VERSION,
    IR_VERSION,
)
from .validator import validate, ValidationResult, ValidationRule
from .whitelist import (
    VERB_WHITELIST,
    VERB_REPLACEMENTS,
    VerbEntry,
    is_approved,
    get_entry,
    list_approved,
    list_excluded,
    suggest_replacement,
    whitelist_report,
)

__all__ = [
    "SilpIR",
    "Entity",
    "Constraint",
    "Alternative",
    "Meta",
    "ACTION_CODE_RE",
    "SILP_VERSION",
    "IR_VERSION",
    "validate",
    "ValidationResult",
    "ValidationRule",
    "VERB_WHITELIST",
    "VERB_REPLACEMENTS",
    "VerbEntry",
    "is_approved",
    "get_entry",
    "list_approved",
    "list_excluded",
    "suggest_replacement",
    "whitelist_report",
]
