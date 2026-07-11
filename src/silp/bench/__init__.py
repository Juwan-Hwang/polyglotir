"""silp-bench — Layers 4 & 5: optimization + migration screening.

Phase 1: A/B test framework for IR granularity and container format.
Phase 4: fitness function and GA machinery (spec §2 Layers 4–5).
"""

from .ab_test import (
    Variant,
    VariantResult,
    GRANULARITIES,
    CONTAINERS,
    generate_variants,
    compile_variant,
    run_variant_matrix,
)

__all__ = [
    "Variant",
    "VariantResult",
    "GRANULARITIES",
    "CONTAINERS",
    "generate_variants",
    "compile_variant",
    "run_variant_matrix",
]
