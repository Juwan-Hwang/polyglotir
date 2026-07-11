"""Tests for Phase 0 modules: JSON frontend, new test cases, judge, models, smoke runner.

These tests verify the full Phase 0 pipeline without requiring network access
or large model downloads.  They use:
- IR schema + validator (no external deps)
- Code / Natural / JSON frontends (no external deps)
- RuleJudge (no API calls)
- Model factory (no model loading)
- Smoke test --dry-run (no model calls)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"


# ── JSON Frontend ─────────────────────────────────────────────────────


class TestJSONFrontend:
    """Test the JSON-slot control baseline frontend."""

    def test_registered(self):
        from silp.frontend import list_frontends
        assert "json" in list_frontends()

    def test_compile_basic(self):
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = {
            "silp": "v1", "version": "ir-v0.1", "intent": "!CANCEL",
            "entities": [{"id": "act", "value": "flight", "action": "!CANCEL"}],
            "constraints": [], "alternatives": [],
            "meta": {"req_id": "a3f9"},
        }
        ir = validate_ir(data).ir
        fe = get_frontend("json")
        output = fe.compile(ir)
        parsed = json.loads(output)
        assert parsed["intent"] == "!CANCEL"
        assert parsed["act"] == "flight"

    def test_compile_with_constraints(self):
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = {
            "silp": "v1", "version": "ir-v0.1", "intent": "!START",
            "entities": [{"id": "act", "value": "hike", "action": "!START"}],
            "constraints": [{"type": "!rain", "value": "true", "time": "t+1"}],
            "alternatives": [{"action": "!START", "target": "cards",
                              "location": "indoor"}],
            "meta": {"req_id": "b7e2"},
        }
        ir = validate_ir(data).ir
        fe = get_frontend("json")
        output = fe.compile(ir)
        parsed = json.loads(output)
        assert parsed["intent"] == "!START"
        assert parsed["act"] == "hike"
        assert parsed["constraints"][0]["type"] == "!rain"
        assert parsed["alternatives"][0]["target"] == "cards"


# ── New test cases (6–10) ─────────────────────────────────────────────


class TestNewTestCases:
    """Verify that cases 6–10 are valid IR and compile correctly."""

    @pytest.mark.parametrize("case_file", [
        "case6_nested_constraint.json",
        "case7_parallel_action.json",
        "case8_conditional_branch.json",
        "case9_tool_call.json",
        "case10_multi_turn.json",
    ])
    def test_ir_valid(self, case_file):
        """Each new case must pass IR validation."""
        from silp.ir import validate as validate_ir

        path = EXAMPLES_DIR / case_file
        assert path.exists(), f"Missing {case_file}"
        data = json.loads(path.read_text(encoding="utf-8"))
        result = validate_ir(data)
        assert result.valid, f"{case_file}: {result.errors}"

    def test_case6_nested_compiles_code(self):
        """Case 6: nested constraints → code with multiple conditions."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads(
            (EXAMPLES_DIR / "case6_nested_constraint.json").read_text("utf-8")
        )
        ir = validate_ir(data).ir
        output = get_frontend("code").compile(ir)
        # Should have budget<=500 and rating>=4.0
        assert "budget<=500" in output
        assert "rating>=4.0" in output
        assert "book(" in output

    def test_case7_parallel_compiles_code(self):
        """Case 7: parallel actions → fetch; process; email."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads(
            (EXAMPLES_DIR / "case7_parallel_action.json").read_text("utf-8")
        )
        ir = validate_ir(data).ir
        output = get_frontend("code").compile(ir)
        assert "fetch(" in output
        assert "process(" in output
        assert "email(" in output
        assert ";" in output  # multiple actions joined

    def test_case8_branch_compiles_code(self):
        """Case 8: conditional branch → route with else alternatives."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads(
            (EXAMPLES_DIR / "case8_conditional_branch.json").read_text("utf-8")
        )
        ir = validate_ir(data).ir
        output = get_frontend("code").compile(ir)
        assert "route(" in output
        assert "else:" in output

    def test_case9_tool_call_compiles_code(self):
        """Case 9: tool call → search with keyword args."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads(
            (EXAMPLES_DIR / "case9_tool_call.json").read_text("utf-8")
        )
        ir = validate_ir(data).ir
        output = get_frontend("code").compile(ir)
        assert "search(" in output
        assert "query=italian" in output

    def test_case10_multi_turn_compiles_code(self):
        """Case 10: multi-turn reference → update with auth constraint."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = json.loads(
            (EXAMPLES_DIR / "case10_multi_turn.json").read_text("utf-8")
        )
        ir = validate_ir(data).ir
        output = get_frontend("code").compile(ir)
        assert "update(" in output
        assert "auth" in output
        assert "status!=shipped" in output


# ── Rule Judge ────────────────────────────────────────────────────────


class TestRuleJudge:
    """Test the rule-based judge (no API needed)."""

    def _make_ir(self):
        from silp.ir import validate as validate_ir

        data = {
            "silp": "v1", "version": "ir-v0.1", "intent": "!CANCEL",
            "entities": [{"id": "act", "value": "flight", "action": "!CANCEL"}],
            "constraints": [], "alternatives": [],
            "meta": {"req_id": "a3f9"},
        }
        return validate_ir(data).ir

    def test_pass_when_response_has_verb_and_entity(self):
        from silp.bench.judge import RuleJudge

        judge = RuleJudge()
        ir = self._make_ir()
        result = judge.judge(ir, "cancel(flight)", "I will cancel the flight.")
        assert result.verdict == "pass"

    def test_fail_when_missing_entity(self):
        from silp.bench.judge import RuleJudge

        judge = RuleJudge()
        ir = self._make_ir()
        result = judge.judge(ir, "cancel(flight)", "I will cancel it.")
        assert result.verdict == "fail"
        assert "missing entities" in result.reason

    def test_fail_when_missing_verb(self):
        from silp.bench.judge import RuleJudge

        judge = RuleJudge()
        ir = self._make_ir()
        result = judge.judge(ir, "cancel(flight)", "The flight is booked.")
        assert result.verdict == "fail"

    def test_pass_with_synonym(self):
        from silp.bench.judge import RuleJudge

        judge = RuleJudge()
        ir = self._make_ir()
        # "void" is a synonym for "cancel"
        result = judge.judge(ir, "cancel(flight)", "I will void the flight.")
        assert result.verdict == "pass"

    def test_negation_logic(self):
        """Rule judge should flag responses that reverse negation."""
        from silp.bench.judge import RuleJudge
        from silp.ir import validate as validate_ir

        data = {
            "silp": "v1", "version": "ir-v0.1", "intent": "!START",
            "entities": [{"id": "act", "value": "hike", "action": "!START"}],
            "constraints": [{"type": "!rain", "value": "true", "time": "t+1"}],
            "alternatives": [],
            "meta": {"req_id": "b7e2"},
        }
        ir = validate_ir(data).ir
        judge = RuleJudge()
        # Response implies it WILL rain and still hikes — wrong
        result = judge.judge(ir, "if !rain(t+1): start(hike)",
                            "It will rain tomorrow, so I will start the hike.")
        assert result.verdict == "fail"


# ── Model factory ─────────────────────────────────────────────────────


class TestModelFactory:
    """Test model lookup (no model loading)."""

    def test_list_models(self):
        from silp.bench.models import list_models

        models = list_models()
        assert "smollm-360m" in models
        assert "deepseek-v3.2" in models
        assert "glm-5.2" in models

    def test_list_model_names_sorted(self):
        from silp.bench.models import list_model_names

        names = list_model_names()
        assert names == sorted(names)
        assert len(names) >= 10  # at least 11 proxy + 3 local + 3 official

    def test_get_local_model(self):
        from silp.bench.models import get_model, LocalHFBackend

        model = get_model("smollm-360m")
        assert isinstance(model, LocalHFBackend)
        assert model.name == "smollm-360m"
        assert model.backend_type == "local"

    def test_get_proxy_model(self):
        from silp.bench.models import get_model, OpenAIBackend

        model = get_model("deepseek-v3.2")
        assert isinstance(model, OpenAIBackend)
        assert model.backend_type == "api"
        assert model.model_id == "deepseek-v3.2"

    def test_get_glm_model(self):
        from silp.bench.models import get_model, OpenAIBackend

        model = get_model("glm-5.2")
        assert isinstance(model, OpenAIBackend)
        assert model.model_id == "glm-5.2"

    def test_get_model_family(self):
        from silp.bench.models import get_model_family

        assert get_model_family("deepseek-v3.2") == "deepseek"
        assert get_model_family("glm-5.2") == "glm"
        assert get_model_family("kimi-k2.6") == "kimi"
        assert get_model_family("smollm-360m") == "smollm"

    def test_get_unknown_model(self):
        from silp.bench.models import get_model

        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent-model")


# ── Judge factory ─────────────────────────────────────────────────────


class TestJudgeFactory:
    """Test judge factory."""

    def test_get_rule_judge(self):
        from silp.bench.judge import get_judge, RuleJudge

        judge = get_judge("rule")
        assert isinstance(judge, RuleJudge)

    def test_get_llm_judge(self):
        from silp.bench.judge import get_judge, LLMJudge

        # This creates the judge object but doesn't call the API
        judge = get_judge("llm", "deepseek-v3.2")
        assert isinstance(judge, LLMJudge)


# ── Tokenizer census (tiktoken only, no HF) ──────────────────────────


class TestTokenizerCensus:
    """Test the census with tiktoken only (no HF model download)."""

    def test_census_verbs(self):
        """Run the verb census with tiktoken only."""
        # This test requires tiktoken to be installed
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken not installed")

        sys.path.insert(0, str(ROOT / "scripts"))
        from tokenizer_census import build_tokenizers, run_census, CENSUS_VERBS

        tokenizers = build_tokenizers(include_hf=False)
        assert len(tokenizers) >= 1

        rows = run_census(CENSUS_VERBS[:5], tokenizers)
        assert len(rows) == 5 * len(tokenizers)

        # Check CSV structure
        for row in rows:
            assert "string" in row
            assert "tokenizer" in row
            assert "token_count" in row
            assert "is_single_token" in row


# ── Smoke test dry run ────────────────────────────────────────────────


class TestSmokeTestDryRun:
    """Test the smoke test runner in dry-run mode (no model calls)."""

    def test_dry_run(self):
        """Dry run should list the matrix without calling models."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_smoke_test import run_smoke_test

        # Dry run with rule judge — no models needed
        run_smoke_test(
            models=["smollm-360m"],  # Will be looked up but not loaded
            frontends=["code", "natural", "json"],
            judge_mode="rule",
            dry_run=True,
        )
        # If we get here without exception, the test passes

    def test_dry_run_all_frontends(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_smoke_test import run_smoke_test
        from silp.frontend import list_frontends

        run_smoke_test(
            models=["qwen2.5-0.5b"],
            frontends=list_frontends(),
            judge_mode="rule",
            dry_run=True,
        )


# ── All 10 cases compile with all frontends ──────────────────────────


class TestAllCasesAllFrontends:
    """Every case must compile successfully with every registered frontend."""

    @pytest.mark.parametrize("case_file", [
        f for f in [
            "case1_multi_constraint.json",
            "case2_negation.json",
            "case3_detail.json",
            "case5_tool_branch.json",
            "case6_nested_constraint.json",
            "case7_parallel_action.json",
            "case8_conditional_branch.json",
            "case9_tool_call.json",
            "case10_multi_turn.json",
        ]
    ])
    def test_compiles_all_frontends(self, case_file):
        from silp.frontend import get_frontend, list_frontends
        from silp.ir import validate as validate_ir

        path = EXAMPLES_DIR / case_file
        data = json.loads(path.read_text(encoding="utf-8"))
        ir = validate_ir(data).ir

        for fe_name in list_frontends():
            fe = get_frontend(fe_name)
            output = fe.compile(ir)
            assert isinstance(output, str)
            assert len(output) > 0, f"{case_file}/{fe_name}: empty output"
