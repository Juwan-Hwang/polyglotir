"""A/B test framework — IR granularity and container format variations.

Phase 1 spec: "A/B 测粒度与容器"

This module generates **variants** of a base IR by varying:

1. **Granularity** — how much detail is in the IR:
   - ``coarse``: Only intent + primary entity (strip constraints, alternatives)
   - ``medium``: Intent + entities + constraints (strip alternatives)
   - ``full``: Complete IR (all fields, as-is)

2. **Container** — how the IR is structured:
   - ``nested``: Standard nested JSON (entities/constraints/alternatives as arrays)
   - ``flat``: Flatten entities into top-level key-value pairs
   - ``compact``: Short field names + minimal separators

Each variant is compiled with every registered frontend, and the token count
is measured across all tokenizers.  This provides a quantitative basis for
Phase 2 model evaluation: which granularity × container × frontend combination
yields the best tradeoff between information density and cross-model
comprehension.

Usage::

    from silp.bench.ab_test import generate_variants, VariantMatrix

    matrix = generate_variants(base_ir)
    for variant in matrix:
        print(variant.label, variant.token_count_gpt4o)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from ..ir.schema import SilpIR, Entity, Constraint, Alternative, Meta


# ── Variant metadata ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Variant:
    """A single IR variant in the A/B test matrix.

    Attributes:
        label: Human-readable identifier (e.g. ``"coarse|flat"``).
        granularity: ``"coarse"``, ``"medium"``, or ``"full"``.
        container: ``"nested"``, ``"flat"``, or ``"compact"``.
        ir: The modified :class:`SilpIR` for this variant.
        description: What was changed from the base IR.
    """

    label: str
    granularity: str
    container: str
    ir: SilpIR
    description: str


# ── Granularity transforms ───────────────────────────────────────────


def to_coarse(ir: SilpIR) -> SilpIR:
    """Strip constraints and alternatives, keep only intent + primary entity."""
    primary_entities = [
        e for e in ir.entities
        if e.action is None or e.action == ir.intent
    ]
    return SilpIR(
        intent=ir.intent,
        entities=primary_entities,
        constraints=[],
        alternatives=[],
        meta=ir.meta,
    )


def to_medium(ir: SilpIR) -> SilpIR:
    """Keep constraints but strip alternatives."""
    return SilpIR(
        intent=ir.intent,
        entities=ir.entities,
        constraints=ir.constraints,
        alternatives=[],
        meta=ir.meta,
    )


# ── Container transforms ─────────────────────────────────────────────


def to_flat(ir: SilpIR) -> str:
    """Render IR as a flat key-value JSON (not nested arrays)."""
    slots: dict[str, object] = {"intent": ir.intent}

    for e in ir.entities:
        if e.action is None or e.action == ir.intent:
            slots[e.id] = e.value
        else:
            key = f"action_{e.action[1:].lower()}"
            slots.setdefault(key, []).append(e.value)

    for i, c in enumerate(ir.constraints):
        prefix = f"c{i}"
        slots[f"{prefix}_type"] = c.type
        slots[f"{prefix}_val"] = c.value
        if c.time:
            slots[f"{prefix}_time"] = c.time

    for i, a in enumerate(ir.alternatives):
        prefix = f"alt{i}"
        slots[f"{prefix}_action"] = a.action
        if a.target:
            slots[f"{prefix}_target"] = a.target
        if a.location:
            slots[f"{prefix}_loc"] = a.location

    return json.dumps(slots, ensure_ascii=False, separators=(",", ":"))


def to_compact(ir: SilpIR) -> str:
    """Render IR with short field names and minimal structure.

    Field name mapping:
        intent → i, entities → e, constraints → k, alternatives → a
        id → d, value → v, action → x, type → t, time → m
        target → g, location → l
    """
    def _entity(e: Entity) -> dict:
        d = {"d": e.id, "v": e.value}
        if e.action:
            d["x"] = e.action
        return d

    def _constraint(c: Constraint) -> dict:
        d = {"t": c.type, "v": c.value}
        if c.time:
            d["m"] = c.time
        operator = getattr(c, "operator", None)
        if operator:
            d["o"] = operator
        subject = getattr(c, "subject", None)
        if subject:
            d["s"] = subject
        return d

    def _alternative(a: Alternative) -> dict:
        d = {"x": a.action}
        if a.target:
            d["g"] = a.target
        if a.location:
            d["l"] = a.location
        return d

    compact = {
        "i": ir.intent,
        "e": [_entity(e) for e in ir.entities],
    }
    if ir.constraints:
        compact["k"] = [_constraint(c) for c in ir.constraints]
    if ir.alternatives:
        compact["a"] = [_alternative(a) for a in ir.alternatives]

    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


# ── Variant matrix generation ───────────────────────────────────────


# Granularity levels
GRANULARITIES = ("coarse", "medium", "full")

# Container formats
CONTAINERS = ("nested", "flat", "compact")


def generate_variants(ir: SilpIR) -> list[Variant]:
    """Generate all granularity × container variants of *ir*.

    Returns a list of :class:`Variant` objects.  For ``nested`` container,
    the variant's ``ir`` is a modified :class:`SilpIR`.  For ``flat`` and
    ``compact``, the variant stores the original IR but the ``compiled_flat``
    or ``compiled_compact`` string is available via :func:`compile_variant`.
    """
    variants: list[Variant] = []

    for granularity in GRANULARITIES:
        # Apply granularity transform
        if granularity == "coarse":
            mod_ir = to_coarse(ir)
            desc = "stripped constraints + alternatives"
        elif granularity == "medium":
            mod_ir = to_medium(ir)
            desc = "stripped alternatives"
        else:  # full
            mod_ir = ir
            desc = "full IR (no changes)"

        for container in CONTAINERS:
            label = f"{granularity}|{container}"
            variants.append(Variant(
                label=label,
                granularity=granularity,
                container=container,
                ir=mod_ir,
                description=desc,
            ))

    return variants


def compile_variant(variant: Variant, frontend_name: str) -> str:
    """Compile a variant with the given frontend.

    For ``nested`` container, uses the frontend's ``compile()`` method.
    For ``flat`` and ``compact``, uses the custom serializers directly.
    """
    if variant.container == "nested":
        from ..frontend import get_frontend
        fe = get_frontend(frontend_name)
        return fe.compile(variant.ir)
    elif variant.container == "flat":
        return to_flat(variant.ir)
    elif variant.container == "compact":
        return to_compact(variant.ir)
    else:
        raise ValueError(f"Unknown container: {variant.container}")


# ── Matrix runner ────────────────────────────────────────────────────


@dataclass
class VariantResult:
    """Token-count results for a single variant × frontend × tokenizer."""

    variant_label: str
    frontend: str
    tokenizer: str
    token_count: int
    compiled_text: str
    char_count: int


def run_variant_matrix(
    ir: SilpIR,
    tokenizers: list | None = None,
    frontend_names: list[str] | None = None,
) -> list[VariantResult]:
    """Run the full variant matrix and measure token counts.

    Args:
        ir: Base IR to generate variants from.
        tokenizers: List of tokenizer objects (with ``encode()`` method).
            If None, only character counts are measured.
        frontend_names: Frontend names to test (default: all registered).

    Returns:
        List of :class:`VariantResult` for each variant × frontend × tokenizer.
    """
    if frontend_names is None:
        from ..frontend import list_frontends
        frontend_names = list_frontends()

    variants = generate_variants(ir)
    results: list[VariantResult] = []

    for variant in variants:
        for fe_name in frontend_names:
            compiled = compile_variant(variant, fe_name)

            if tokenizers:
                for tok in tokenizers:
                    try:
                        ids = tok.encode(compiled)
                        tc = len(ids)
                    except Exception:
                        tc = -1
                    results.append(VariantResult(
                        variant_label=variant.label,
                        frontend=fe_name,
                        tokenizer=tok.name,
                        token_count=tc,
                        compiled_text=compiled,
                        char_count=len(compiled),
                    ))
            else:
                results.append(VariantResult(
                    variant_label=variant.label,
                    frontend=fe_name,
                    tokenizer="char_count",
                    token_count=len(compiled),
                    compiled_text=compiled,
                    char_count=len(compiled),
                ))

    return results


__all__ = [
    "Variant",
    "VariantResult",
    "GRANULARITIES",
    "CONTAINERS",
    "generate_variants",
    "compile_variant",
    "run_variant_matrix",
    "to_coarse",
    "to_medium",
    "to_flat",
    "to_compact",
]
