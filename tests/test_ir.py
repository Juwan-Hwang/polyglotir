"""Tests for the IR schema and validator.

Covers the spec's canonical example (§2 Layer 1) plus edge cases:
- Action code format validation
- Meta extension fields (silently allowed)
- req_id format
- Root-level extra fields (forbidden)
"""

from __future__ import annotations

import pytest

from silp.ir import (
    SilpIR,
    Entity,
    Constraint,
    Alternative,
    Meta,
    validate,
    ACTION_CODE_RE,
)


# ── Fixtures ──────────────────────────────────────────────────────────

# Canonical example from the spec (§2 Layer 1)
CANONICAL_IR = {
    "silp": "v1",
    "version": "ir-v0.1",
    "intent": "!CANCEL",
    "entities": [{"id": "act", "value": "flight", "action": "!CANCEL"}],
    "constraints": [{"type": "weather", "value": "severe_rain", "time": "t+1"}],
    "alternatives": [{"action": "!START", "target": "cards", "location": "indoor"}],
    "meta": {
        "priority": 1,
        "confidence": 0.9,
        "seq": ["reason_first"],
        "out": "natural",
        "next_agent": None,
        "req_id": "a3f9",
        "session_id": None,
    },
}


# ── Schema tests ──────────────────────────────────────────────────────


class TestActionCodePattern:
    def test_valid_codes(self):
        for code in ["!CANCEL", "!START", "!SWITCH_TOOL", "!EMAIL"]:
            assert ACTION_CODE_RE.match(code)

    def test_invalid_codes(self):
        for code in ["CANCEL", "!cancel", "!Cancel", "!", "!123", "!CANCE L"]:
            assert not ACTION_CODE_RE.match(code)


class TestSilpIRConstruction:
    def test_canonical_example(self):
        ir = SilpIR(**CANONICAL_IR)
        assert ir.intent == "!CANCEL"
        assert ir.entities[0].value == "flight"
        assert ir.meta.req_id == "a3f9"

    def test_intent_must_be_action_code(self):
        bad = {**CANONICAL_IR, "intent": "CANCEL"}
        with pytest.raises(Exception, match="intent"):
            SilpIR(**bad)

    def test_root_extra_forbidden(self):
        bad = {**CANONICAL_IR, "unknown_field": 42}
        with pytest.raises(Exception):
            SilpIR(**bad)

    def test_meta_extra_allowed(self):
        """Extension fields in meta must be silently accepted."""
        data = {**CANONICAL_IR}
        data["meta"] = {**data["meta"], "custom_field": "anything"}
        ir = SilpIR(**data)
        assert ir.meta.custom_field == "anything"  # type: ignore[attr-defined]


class TestReqId:
    def test_valid_4_hex(self):
        ir = SilpIR(**CANONICAL_IR)
        assert ir.meta.req_id == "a3f9"

    def test_invalid_short(self):
        data = {**CANONICAL_IR, "meta": {**CANONICAL_IR["meta"], "req_id": "ab"}}
        with pytest.raises(Exception, match="req_id"):
            SilpIR(**data)

    def test_invalid_non_hex(self):
        data = {**CANONICAL_IR, "meta": {**CANONICAL_IR["meta"], "req_id": "xyzw"}}
        with pytest.raises(Exception, match="req_id"):
            SilpIR(**data)

    def test_generate(self):
        rid = SilpIR.generate_req_id("test content")
        assert len(rid) == 4
        assert all(c in "0123456789abcdef" for c in rid)


# ── Validator tests ───────────────────────────────────────────────────


class TestValidator:
    def test_valid_canonical(self):
        result = validate(CANONICAL_IR)
        assert result.valid
        assert result.ir is not None
        assert result.ir.intent == "!CANCEL"

    def test_invalid_intent(self):
        bad = {**CANONICAL_IR, "intent": "not_a_code"}
        result = validate(bad)
        assert not result.valid
        assert any("intent" in e for e in result.errors)

    def test_secondary_action_warning(self):
        """Entity with different action → warning, not error."""
        data = {**CANONICAL_IR}
        data["entities"] = [
            {"id": "act", "value": "flight", "action": "!CANCEL"},
            {"id": "notify", "value": "zhangsan", "action": "!EMAIL"},
        ]
        result = validate(data)
        assert result.valid
        assert any("secondary action" in w for w in result.warnings)

    def test_missing_required_field(self):
        bad = {**CANONICAL_IR}
        del bad["intent"]
        result = validate(bad)
        assert not result.valid

    def test_result_bool(self):
        """ValidationResult supports `if result:` syntax."""
        assert bool(validate(CANONICAL_IR))
        bad = {**CANONICAL_IR, "intent": "bad"}
        assert not bool(validate(bad))
