"""Tests for the code frontend compiler.

Uses the five canonical test cases from the spec (§附 Phase 0.5 execution list)
to verify that IR → code-frontend compilation produces the expected surface strings.
"""

from __future__ import annotations

import pytest

from silp.frontend import CodeFrontend, NaturalFrontend, get_frontend, list_frontends
from silp.ir import SilpIR


# ── Helpers ───────────────────────────────────────────────────────────


def _make_ir(**overrides) -> SilpIR:
    """Build a SilpIR from the canonical template with overrides."""
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
    return SilpIR(**base)


# ── Registry tests ────────────────────────────────────────────────────


class TestRegistry:
    def test_code_frontend_registered(self):
        assert "code" in list_frontends()

    def test_natural_frontend_registered(self):
        assert "natural" in list_frontends()

    def test_get_unknown_frontend(self):
        with pytest.raises(KeyError, match="Unknown frontend"):
            get_frontend("nonexistent")


# ── Code frontend: spec test cases ────────────────────────────────────


class TestCodeFrontendCompile:
    fe = CodeFrontend()

    def test_case1_multi_constraint_logic(self):
        """if loc(me,Beijing,t+1am): cancel(flight,t+1pm); email(zhangsan)"""
        ir = _make_ir(
            intent="!CANCEL",
            entities=[
                {"id": "act", "value": "flight", "action": "!CANCEL"},
                {"id": "notify", "value": "zhangsan", "action": "!EMAIL"},
            ],
            constraints=[
                {"type": "loc", "value": "Beijing", "time": "t+1am",
                 "subject": "me"},
            ],
        )
        output = self.fe.compile(ir)
        assert "if loc(me,Beijing,t+1am):" in output
        assert "cancel(flight)" in output
        assert "email(zhangsan)" in output

    def test_case2_negation_logic(self):
        """if !rain(t+1): start(hike) else: start(cards@indoor)"""
        ir = _make_ir(
            intent="!START",
            entities=[{"id": "act", "value": "hike", "action": "!START"}],
            constraints=[
                {"type": "!rain", "value": "true", "time": "t+1"},
            ],
            alternatives=[
                {"action": "!START", "target": "cards", "location": "indoor"},
            ],
        )
        output = self.fe.compile(ir)
        assert "if !rain(t+1):" in output
        assert "start(hike)" in output
        assert "else:" in output
        assert "start(cards@indoor)" in output

    def test_case3_detail_preservation(self):
        """translate(src=fr_rev_bold, tgt=shakespeare_en, style=archaic_heavy)"""
        ir = _make_ir(
            intent="!TRANSLATE",
            entities=[
                {"id": "src", "value": "fr_rev_bold"},
                {"id": "tgt", "value": "shakespeare_en"},
                {"id": "style", "value": "archaic_heavy"},
            ],
            constraints=[],
        )
        output = self.fe.compile(ir)
        assert "translate(src=fr_rev_bold,tgt=shakespeare_en,style=archaic_heavy)" in output

    def test_case5_tool_call_branch(self):
        """if weather>rain: switch_tool(indoor_activity)"""
        ir = _make_ir(
            intent="!SWITCH_TOOL",
            entities=[{"id": "act", "value": "indoor_activity", "action": "!SWITCH_TOOL"}],
            constraints=[
                {"type": "weather", "value": "rain", "operator": ">"},
            ],
        )
        output = self.fe.compile(ir)
        assert "if weather>rain:" in output
        assert "switch_tool(indoor_activity)" in output

    def test_no_constraints_no_alternatives(self):
        """Bare function call when no conditions or alternatives."""
        ir = _make_ir(constraints=[], alternatives=[])
        output = self.fe.compile(ir)
        assert output == "cancel(flight)"


# ── Natural frontend tests ────────────────────────────────────────────


class TestNaturalFrontend:
    fe = NaturalFrontend()

    def test_produces_readable_prose(self):
        ir = _make_ir()
        output = self.fe.compile(ir)
        assert "cancel" in output.lower()
        assert "flight" in output.lower()

    def test_decode_not_supported(self):
        with pytest.raises(NotImplementedError):
            self.fe.decode("some text")

    def test_negation_in_prose(self):
        ir = _make_ir(
            intent="!START",
            entities=[{"id": "act", "value": "hike", "action": "!START"}],
            constraints=[{"type": "!rain", "value": "true", "time": "t+1"}],
            alternatives=[
                {"action": "!START", "target": "cards", "location": "indoor"},
            ],
        )
        output = self.fe.compile(ir)
        assert "not rain" in output.lower()
        assert "otherwise" in output.lower()
