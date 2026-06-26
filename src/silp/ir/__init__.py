"""silp-ir — Layer 1: Semantic Intermediate Representation.

Provides the coarse-grained task-slot IR (JSON Schema via Pydantic v2),
a built-in validator, and the verb whitelist machinery.
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
from .validator import validate, ValidationResult, VERB_WHITELIST

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
    "VERB_WHITELIST",
]
