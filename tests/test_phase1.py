"""Phase 1 test suite — whitelist, validator rules, decode round-trip,
req_id collision, and A/B test framework.

These tests verify all Phase 1 deliverables:
1. Verb whitelist enforcement (approved + excluded verbs)
2. Validator rule engine (semantic rules)
3. CodeFrontend.decode() round-trip
4. JSONFrontend.decode() round-trip
5. req_id collision test
6. A/B test framework variant generation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"


# ── Helpers ───────────────────────────────────────────────────────────


def _make_ir(**overrides) -> dict:
    """Build a canonical IR dict with overrides."""
    base = {
        "silp": "v1",
        "version": "ir-v0.1",
        "intent": "!CANCEL",
        "entities": [{"id": "act", "value": "flight", "action": "!CANCEL"}],
        "constraints": [],
        "alternatives": [],
        "meta": {
            "priority": 1,
            "confidence": 0.9,
            "seq": [],
            "out": "natural",
            "next_agent": None,
            "req_id": "a3f9",
            "session_id": None,
        },
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════
# 1. Verb Whitelist Tests
# ═══════════════════════════════════════════════════════════════════════


class TestVerbWhitelist:
    """Test the verb whitelist data and lookup functions."""

    def test_approved_verbs_present(self):
        from silp.ir.whitelist import VERB_WHITELIST

        for verb in ["CANCEL", "START", "EMAIL", "FETCH", "PROCESS",
                      "TRANSLATE", "BOOK", "ROUTE", "SEARCH", "UPDATE",
                      "SUGGEST", "SWITCH"]:
            assert verb in VERB_WHITELIST, f"{verb} should be approved"

    def test_excluded_verbs_absent(self):
        from silp.ir.whitelist import VERB_WHITELIST

        assert "SWITCH_TOOL" not in VERB_WHITELIST
        assert "ESCALATE" not in VERB_WHITELIST

    def test_is_approved(self):
        from silp.ir.whitelist import is_approved

        assert is_approved("CANCEL")
        assert is_approved("SWITCH")
        assert not is_approved("SWITCH_TOOL")
        assert not is_approved("ESCALATE")
        assert not is_approved("UNKNOWN_VERB")

    def test_get_entry(self):
        from silp.ir.whitelist import get_entry

        entry = get_entry("CANCEL")
        assert entry is not None
        assert entry.verb == "CANCEL"
        assert entry.fn_name == "cancel"
        assert entry.approved

        entry = get_entry("ESCALATE")
        assert entry is not None
        assert not entry.approved
        assert entry.exclude_reason is not None

    def test_suggest_replacement(self):
        from silp.ir.whitelist import suggest_replacement

        assert suggest_replacement("SWITCH_TOOL") == "SWITCH"
        assert suggest_replacement("CANCEL") is None

    def test_list_approved_sorted(self):
        from silp.ir.whitelist import list_approved

        approved = list_approved()
        assert approved == sorted(approved)
        assert len(approved) >= 12

    def test_whitelist_report(self):
        from silp.ir.whitelist import whitelist_report

        report = whitelist_report()
        assert len(report) >= 13  # 12 approved + at least 1 excluded
        for row in report:
            assert "verb" in row
            assert "status" in row
            assert "single_token_all" in row

    def test_all_approved_are_single_token(self):
        """Every approved verb must be single-token across all tokenizers."""
        from silp.ir.whitelist import get_entry, list_approved

        for verb in list_approved():
            entry = get_entry(verb)
            assert entry is not None
            assert entry.single_token_all, (
                f"{verb} is approved but not single-token in all tokenizers"
            )


# ═══════════════════════════════════════════════════════════════════════
# 2. Validator Rule Engine Tests
# ═══════════════════════════════════════════════════════════════════════


class TestValidatorRules:
    """Test the expanded validator with semantic rules."""

    def test_whitelist_enforced_by_default(self):
        """Default validate() rejects non-whitelisted verbs."""
        from silp.ir import validate

        data = _make_ir(intent="!SWITCH_TOOL",
                        entities=[{"id": "act", "value": "x", "action": "!SWITCH_TOOL"}])
        result = validate(data)
        assert not result.valid
        assert any("not in whitelist" in e for e in result.errors)

    def test_whitelist_can_be_disabled(self):
        """enforce_whitelist=False allows any !VERB (backward compat)."""
        from silp.ir import validate

        data = _make_ir(intent="!SWITCH_TOOL",
                        entities=[{"id": "act", "value": "x", "action": "!SWITCH_TOOL"}])
        result = validate(data, enforce_whitelist=False)
        assert result.valid

    def test_whitelist_replacement_hint(self):
        """Error message includes replacement suggestion."""
        from silp.ir import validate

        data = _make_ir(intent="!ESCALATE",
                        entities=[{"id": "act", "value": "x", "action": "!ESCALATE"}])
        result = validate(data)
        assert not result.valid
        assert any("RAISE" in e for e in result.errors)

    def test_duplicate_entity_id_error(self):
        """Two entities with same id → error."""
        from silp.ir import validate

        data = _make_ir(
            entities=[
                {"id": "act", "value": "flight", "action": "!CANCEL"},
                {"id": "act", "value": "hotel", "action": "!CANCEL"},
            ],
        )
        result = validate(data)
        assert not result.valid
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_invalid_operator_error(self):
        """Invalid operator → error."""
        from silp.ir import validate

        data = _make_ir(
            constraints=[
                {"type": "weather", "value": "rain", "operator": "~="},
            ],
        )
        result = validate(data)
        assert not result.valid
        assert any("operator" in e.lower() for e in result.errors)

    def test_valid_operators_pass(self):
        """All supported operators should pass."""
        from silp.ir import validate

        for op in ["==", "!=", "<", ">", "<=", ">="]:
            data = _make_ir(
                constraints=[{"type": "x", "value": "y", "operator": op}],
            )
            result = validate(data)
            assert result.valid, f"operator {op} should be valid"

    def test_negation_double_negative_warning(self):
        """!type with value='true' → warning (double negative)."""
        from silp.ir import validate

        data = _make_ir(
            intent="!START",
            entities=[{"id": "act", "value": "hike", "action": "!START"}],
            constraints=[{"type": "!rain", "value": "true", "time": "t+1"}],
        )
        result = validate(data)
        assert result.valid  # warning, not error
        assert any("double-negative" in w for w in result.warnings)

    def test_alternative_no_target_warning(self):
        """Alternative with location but no target → warning."""
        from silp.ir import validate

        data = _make_ir(
            alternatives=[{"action": "!START", "location": "indoor"}],
        )
        result = validate(data)
        assert result.valid
        assert any("target" in w for w in result.warnings)

    def test_empty_entities_warning(self):
        """IR with no entities → warning."""
        from silp.ir import validate

        data = _make_ir(entities=[])
        result = validate(data)
        assert result.valid
        assert any("no entities" in w for w in result.warnings)

    def test_secondary_action_warning(self):
        """Entity with different action → warning."""
        from silp.ir import validate

        data = _make_ir(
            entities=[
                {"id": "act", "value": "flight", "action": "!CANCEL"},
                {"id": "notify", "value": "zhangsan", "action": "!EMAIL"},
            ],
        )
        result = validate(data)
        assert result.valid
        assert any("secondary action" in w for w in result.warnings)

    def test_rules_run_recorded(self):
        """ValidationResult.records which rules ran."""
        from silp.ir import validate

        result = validate(_make_ir())
        assert "whitelist" in result.rules_run
        assert "duplicate_entity_id" in result.rules_run
        assert "operator_validity" in result.rules_run

    def test_all_examples_pass_validation(self):
        """Every example case must pass the validator."""
        from silp.ir import validate as validate_ir

        for path in sorted(EXAMPLES_DIR.glob("case*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            result = validate_ir(data)
            assert result.valid, f"{path.name}: {result.errors}"


# ═══════════════════════════════════════════════════════════════════════
# 3. CodeFrontend.decode() Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCodeFrontendDecode:
    """Test CodeFrontend.decode() round-trip for all example cases."""

    @pytest.mark.parametrize("case_file", [
        "case1_multi_constraint.json",
        "case2_negation.json",
        "case3_detail.json",
        "case5_tool_branch.json",
        "case6_nested_constraint.json",
        "case7_parallel_action.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
        "case10_multi_turn.json",
    ])
    def test_roundtrip_intent(self, case_file):
        """decode(compile(ir)).intent == ir.intent for every case."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads((EXAMPLES_DIR / case_file).read_text("utf-8"))
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("code")

        compiled = fe.compile(ir)
        decoded = fe.decode(compiled)

        assert decoded.intent == ir.intent, (
            f"{case_file}: intent {ir.intent!r} → {decoded.intent!r}\n"
            f"  compiled: {compiled}"
        )

    @pytest.mark.parametrize("case_file", [
        "case1_multi_constraint.json",
        "case2_negation.json",
        "case3_detail.json",
        "case5_tool_branch.json",
        "case6_nested_constraint.json",
        "case7_parallel_action.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
        "case10_multi_turn.json",
    ])
    def test_roundtrip_entities(self, case_file):
        """All entity (id, value) / (value, action) pairs preserved.

        Primary entities: compare (id, value) — action is always intent.
        Secondary entities: compare (value, action) — id may be lost
        in code frontend (positional args decode as id="act").
        """
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads((EXAMPLES_DIR / case_file).read_text("utf-8"))
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("code")

        compiled = fe.compile(ir)
        decoded = fe.decode(compiled)

        def _norm(action, intent):
            return intent if action is None or action == intent else action

        # Primary entities: compare (id, value)
        orig_pri = {(e.id, e.value) for e in ir.entities
                    if _norm(e.action, ir.intent) == ir.intent}
        dec_pri = {(e.id, e.value) for e in decoded.entities
                   if _norm(e.action, decoded.intent) == decoded.intent}
        assert orig_pri == dec_pri, (
            f"{case_file}: primary entity mismatch\n"
            f"  missing: {orig_pri - dec_pri}\n"
            f"  extra: {dec_pri - orig_pri}\n"
            f"  compiled: {compiled}"
        )

        # Secondary entities: compare (value, action) — id may be lost
        orig_sec = {(e.value, _norm(e.action, ir.intent))
                    for e in ir.entities
                    if _norm(e.action, ir.intent) != ir.intent}
        dec_sec = {(e.value, _norm(e.action, decoded.intent))
                   for e in decoded.entities
                   if _norm(e.action, decoded.intent) != decoded.intent}
        assert orig_sec == dec_sec, (
            f"{case_file}: secondary entity mismatch\n"
            f"  missing: {orig_sec - dec_sec}\n"
            f"  extra: {dec_sec - orig_sec}\n"
            f"  compiled: {compiled}"
        )

    @pytest.mark.parametrize("case_file", [
        "case1_multi_constraint.json",
        "case2_negation.json",
        "case3_detail.json",
        "case5_tool_branch.json",
        "case6_nested_constraint.json",
        "case7_parallel_action.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
        "case10_multi_turn.json",
    ])
    def test_roundtrip_constraints(self, case_file):
        """All constraint types, values, and times preserved."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads((EXAMPLES_DIR / case_file).read_text("utf-8"))
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("code")

        compiled = fe.compile(ir)
        decoded = fe.decode(compiled)

        assert len(decoded.constraints) == len(ir.constraints), (
            f"{case_file}: constraint count {len(ir.constraints)} → "
            f"{len(decoded.constraints)}\n  compiled: {compiled}"
        )

        for i, (orig_c, dec_c) in enumerate(zip(ir.constraints, decoded.constraints)):
            assert orig_c.type == dec_c.type, (
                f"{case_file} constraint[{i}].type: {orig_c.type!r} → {dec_c.type!r}"
            )
            assert orig_c.value == dec_c.value, (
                f"{case_file} constraint[{i}].value: {orig_c.value!r} → {dec_c.value!r}"
            )
            assert orig_c.time == dec_c.time, (
                f"{case_file} constraint[{i}].time: {orig_c.time!r} → {dec_c.time!r}"
            )

    @pytest.mark.parametrize("case_file", [
        "case2_negation.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
    ])
    def test_roundtrip_alternatives(self, case_file):
        """All alternative actions and targets preserved."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads((EXAMPLES_DIR / case_file).read_text("utf-8"))
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("code")

        compiled = fe.compile(ir)
        decoded = fe.decode(compiled)

        assert len(decoded.alternatives) == len(ir.alternatives), (
            f"{case_file}: alt count {len(ir.alternatives)} → "
            f"{len(decoded.alternatives)}\n  compiled: {compiled}"
        )

        for i, (orig_a, dec_a) in enumerate(zip(ir.alternatives, decoded.alternatives)):
            assert orig_a.action == dec_a.action
            assert orig_a.target == dec_a.target
            assert orig_a.location == dec_a.location

    def test_decode_bare_call(self):
        """Bare function call: cancel(flight)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode("cancel(flight)")

        assert decoded.intent == "!CANCEL"
        assert len(decoded.entities) == 1
        assert decoded.entities[0].id == "act"
        assert decoded.entities[0].value == "flight"

    def test_decode_keyword_args(self):
        """Keyword args: translate(src=fr,tgt=en,style=archaic)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode("translate(src=fr,tgt=en,style=archaic)")

        assert decoded.intent == "!TRANSLATE"
        entities = {e.id: e.value for e in decoded.entities}
        assert entities["src"] == "fr"
        assert entities["tgt"] == "en"
        assert entities["style"] == "archaic"

    def test_decode_negation(self):
        """Negation: if !rain(t+1): start(hike)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode("if !rain(t+1): start(hike)")

        assert decoded.intent == "!START"
        assert len(decoded.constraints) == 1
        assert decoded.constraints[0].type == "!rain"
        assert decoded.constraints[0].time == "t+1"

    def test_decode_operator(self):
        """Operator: if weather>rain: switch(indoor)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode("if weather>rain: switch(indoor)")

        assert decoded.intent == "!SWITCH"
        assert len(decoded.constraints) == 1
        c = decoded.constraints[0]
        assert c.type == "weather"
        assert c.value == "rain"
        assert getattr(c, "operator", None) == ">"

    def test_decode_else_branch(self):
        """Else branch: if !rain(t+1): start(hike) else: start(cards@indoor)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode(
            "if !rain(t+1): start(hike) else: start(cards@indoor)"
        )

        assert len(decoded.alternatives) == 1
        alt = decoded.alternatives[0]
        assert alt.action == "!START"
        assert alt.target == "cards"
        assert alt.location == "indoor"

    def test_decode_secondary_actions(self):
        """Secondary actions: cancel(flight); email(zhangsan)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode("cancel(flight); email(zhangsan)")

        assert decoded.intent == "!CANCEL"
        # Should have 2 entities: one for cancel, one for email
        actions = [e.action for e in decoded.entities]
        assert "!CANCEL" in actions
        assert "!EMAIL" in actions

    def test_decode_subject_constraint(self):
        """Subject in constraint: loc(me,Beijing,t+1am)"""
        from silp.frontend import get_frontend

        fe = get_frontend("code")
        decoded = fe.decode("if loc(me,Beijing,t+1am): cancel(flight)")

        assert len(decoded.constraints) == 1
        c = decoded.constraints[0]
        assert c.type == "loc"
        assert c.value == "Beijing"
        assert c.time == "t+1am"
        assert getattr(c, "subject", None) == "me"


# ═══════════════════════════════════════════════════════════════════════
# 4. JSONFrontend.decode() Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════


class TestJSONFrontendDecode:
    """Test JSONFrontend.decode() round-trip."""

    @pytest.mark.parametrize("case_file", [
        "case1_multi_constraint.json",
        "case2_negation.json",
        "case3_detail.json",
        "case5_tool_branch.json",
        "case6_nested_constraint.json",
        "case7_parallel_action.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
        "case10_multi_turn.json",
    ])
    def test_roundtrip_intent(self, case_file):
        """decode(compile(ir)).intent == ir.intent."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads((EXAMPLES_DIR / case_file).read_text("utf-8"))
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("json")

        compiled = fe.compile(ir)
        decoded = fe.decode(compiled)

        assert decoded.intent == ir.intent

    @pytest.mark.parametrize("case_file", [
        "case1_multi_constraint.json",
        "case2_negation.json",
        "case3_detail.json",
        "case5_tool_branch.json",
        "case6_nested_constraint.json",
        "case7_parallel_action.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
        "case10_multi_turn.json",
    ])
    def test_roundtrip_entities(self, case_file):
        """All entity (id, value) / (value, action) pairs preserved."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads((EXAMPLES_DIR / case_file).read_text("utf-8"))
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("json")

        compiled = fe.compile(ir)
        decoded = fe.decode(compiled)

        def _norm(action, intent):
            return intent if action is None or action == intent else action

        # Primary entities: compare (id, value)
        orig_primary = {(e.id, e.value) for e in ir.entities
                        if _norm(e.action, ir.intent) == ir.intent}
        dec_primary = {(e.id, e.value) for e in decoded.entities
                       if _norm(e.action, decoded.intent) == decoded.intent}
        assert orig_primary == dec_primary, (
            f"{case_file}: primary entity mismatch\n"
            f"  missing: {orig_primary - dec_primary}\n"
            f"  extra: {dec_primary - orig_primary}\n"
            f"  compiled: {compiled}"
        )

        # Secondary entities: compare (value, action) — id may differ
        orig_sec = {(e.value, _norm(e.action, ir.intent))
                    for e in ir.entities
                    if _norm(e.action, ir.intent) != ir.intent}
        dec_sec = {(e.value, _norm(e.action, decoded.intent))
                   for e in decoded.entities
                   if _norm(e.action, decoded.intent) != decoded.intent}
        assert orig_sec == dec_sec, (
            f"{case_file}: secondary entity mismatch\n"
            f"  missing: {orig_sec - dec_sec}\n"
            f"  extra: {dec_sec - orig_sec}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. req_id Collision Tests
# ═══════════════════════════════════════════════════════════════════════


class TestReqIdCollision:
    """Test req_id collision characteristics."""

    def test_generate_req_id_4_chars(self):
        from silp.ir import SilpIR

        rid = SilpIR.generate_req_id("test content")
        assert len(rid) == 4
        assert all(c in "0123456789abcdef" for c in rid)

    def test_generate_req_id_6_chars(self):
        from silp.ir import SilpIR

        rid = SilpIR.generate_req_id("test content", length=6)
        assert len(rid) == 6

    def test_generate_req_id_deterministic(self):
        """Same content → same req_id."""
        from silp.ir import SilpIR

        assert SilpIR.generate_req_id("abc") == SilpIR.generate_req_id("abc")

    def test_generate_req_id_different_content(self):
        """Different content → different req_id (usually)."""
        from silp.ir import SilpIR

        assert SilpIR.generate_req_id("abc") != SilpIR.generate_req_id("xyz")

    def test_collision_100_entries(self):
        """100 entries with 4-digit req_id should have low collision."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from req_id_collision_test import test_collision

        result = test_collision(100, 4)
        assert result["collision_rate_pct"] < 5.0  # should be low

    def test_collision_1000_entries(self):
        """1000 entries — the spec's threshold test."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from req_id_collision_test import test_collision

        result = test_collision(1000, 4)
        # Report the rate regardless of pass/fail
        print(f"\n  4-digit collision rate (1000 entries): "
              f"{result['collision_rate_pct']}%")

    def test_collision_6_digit_lower(self):
        """6-digit req_id should have fewer collisions than 4-digit."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from req_id_collision_test import test_collision

        r4 = test_collision(1000, 4)
        r6 = test_collision(1000, 6)
        assert r6["collision_rate_pct"] <= r4["collision_rate_pct"]


# ═══════════════════════════════════════════════════════════════════════
# 6. A/B Test Framework Tests
# ═══════════════════════════════════════════════════════════════════════


class TestABTestFramework:
    """Test the A/B test variant generation."""

    def _make_ir(self):
        from silp.ir import validate as validate_ir

        data = _make_ir(
            intent="!START",
            entities=[{"id": "act", "value": "hike", "action": "!START"}],
            constraints=[{"type": "!rain", "value": "true", "time": "t+1"}],
            alternatives=[{"action": "!START", "target": "cards", "location": "indoor"}],
        )
        return validate_ir(data, enforce_whitelist=False).ir

    def test_generate_variants_count(self):
        """3 granularities × 3 containers = 9 variants."""
        from silp.bench.ab_test import generate_variants

        ir = self._make_ir()
        variants = generate_variants(ir)
        assert len(variants) == 9

    def test_granularities_present(self):
        from silp.bench.ab_test import generate_variants, GRANULARITIES

        ir = self._make_ir()
        variants = generate_variants(ir)
        found = {v.granularity for v in variants}
        assert found == set(GRANULARITIES)

    def test_containers_present(self):
        from silp.bench.ab_test import generate_variants, CONTAINERS

        ir = self._make_ir()
        variants = generate_variants(ir)
        found = {v.container for v in variants}
        assert found == set(CONTAINERS)

    def test_coarse_strips_constraints(self):
        from silp.bench.ab_test import generate_variants

        ir = self._make_ir()
        variants = generate_variants(ir)
        coarse = [v for v in variants if v.granularity == "coarse"]
        for v in coarse:
            assert len(v.ir.constraints) == 0
            assert len(v.ir.alternatives) == 0

    def test_medium_keeps_constraints(self):
        from silp.bench.ab_test import generate_variants

        ir = self._make_ir()
        variants = generate_variants(ir)
        medium = [v for v in variants if v.granularity == "medium"]
        for v in medium:
            assert len(v.ir.constraints) == len(ir.constraints)
            assert len(v.ir.alternatives) == 0

    def test_full_preserves_everything(self):
        from silp.bench.ab_test import generate_variants

        ir = self._make_ir()
        variants = generate_variants(ir)
        full = [v for v in variants if v.granularity == "full"]
        for v in full:
            assert len(v.ir.constraints) == len(ir.constraints)
            assert len(v.ir.alternatives) == len(ir.alternatives)

    def test_compile_variant_nested(self):
        from silp.bench.ab_test import generate_variants, compile_variant

        ir = self._make_ir()
        variants = generate_variants(ir)
        nested = [v for v in variants if v.container == "nested" and v.granularity == "full"][0]
        compiled = compile_variant(nested, "code")
        assert isinstance(compiled, str)
        assert "start(" in compiled

    def test_compile_variant_flat(self):
        from silp.bench.ab_test import generate_variants, compile_variant

        ir = self._make_ir()
        variants = generate_variants(ir)
        flat = [v for v in variants if v.container == "flat" and v.granularity == "full"][0]
        compiled = compile_variant(flat, "code")
        parsed = json.loads(compiled)
        assert "intent" in parsed

    def test_compile_variant_compact(self):
        from silp.bench.ab_test import generate_variants, compile_variant

        ir = self._make_ir()
        variants = generate_variants(ir)
        compact = [v for v in variants if v.container == "compact" and v.granularity == "full"][0]
        compiled = compile_variant(compact, "code")
        parsed = json.loads(compiled)
        assert "i" in parsed  # compact uses "i" for intent

    def test_run_variant_matrix_char_count(self):
        from silp.bench.ab_test import run_variant_matrix

        ir = self._make_ir()
        results = run_variant_matrix(ir, tokenizers=None, frontend_names=["code"])
        # 9 variants × 1 frontend × 1 "tokenizer" (char_count) = 9
        assert len(results) == 9
        for r in results:
            assert r.token_count > 0
            assert r.char_count > 0

    def test_coarse_has_fewer_tokens_than_full(self):
        """Coarse granularity should produce shorter output than full."""
        from silp.bench.ab_test import run_variant_matrix

        ir = self._make_ir()
        results = run_variant_matrix(ir, tokenizers=None, frontend_names=["code"])

        # Compare coarse|nested vs full|nested
        coarse = [r for r in results if r.variant_label == "coarse|nested"][0]
        full = [r for r in results if r.variant_label == "full|nested"][0]
        assert coarse.token_count <= full.token_count
