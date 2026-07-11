"""Phase 2 test suite — retry mechanism, baseline frontends, probes, Spearman.

Tests:
1. Infra-error retry classification (is_infra_error)
2. ModelResponse.retries field
3. NLInJSONFrontend compile
4. LLMLingua2Frontend import + lazy load
5. Frontend registry includes new frontends
6. Phase 2 matrix script dry-run
7. Probe output files exist and have data
8. Spearman correlation computation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EXAMPLES_DIR = ROOT / "examples"
PHASE2_PROCESSED = ROOT / "data" / "processed" / "phase2"


def _make_ir(**overrides) -> dict:
    base = {
        "silp": "v1",
        "version": "ir-v0.1",
        "intent": "!CANCEL",
        "entities": [{"id": "act", "value": "flight", "action": "!CANCEL"}],
        "constraints": [],
        "alternatives": [],
        "meta": {
            "priority": 1, "confidence": 0.9, "seq": [],
            "out": "natural", "next_agent": None,
            "req_id": "a3f9", "session_id": None,
        },
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════
# 1. Infra-Error Retry Classification
# ═══════════════════════════════════════════════════════════════════════


class TestInfraErrorClassification:
    """Test the is_infra_error() classifier — the core of retry policy."""

    def test_timeout_is_infra(self):
        from silp.bench.models import is_infra_error
        assert is_infra_error("Connection timeout after 30s")
        assert is_infra_error("Request timed out")
        assert is_infra_error("read timeout")

    def test_connection_error_is_infra(self):
        from silp.bench.models import is_infra_error
        assert is_infra_error("ConnectionError: connection refused")
        assert is_infra_error("Connection reset by peer")
        assert is_infra_error("broken pipe")

    def test_rate_limit_is_infra(self):
        from silp.bench.models import is_infra_error
        assert is_infra_error("Rate limit exceeded (429)")
        assert is_infra_error("overloaded")

    def test_5xx_is_infra(self):
        from silp.bench.models import is_infra_error
        assert is_infra_error("HTTP 500 internal server error")
        assert is_infra_error("502 bad gateway")
        assert is_infra_error("503 service unavailable")

    def test_empty_response_is_infra(self):
        from silp.bench.models import is_infra_error
        assert is_infra_error("Empty response from model")

    def test_auth_error_not_infra(self):
        from silp.bench.models import is_infra_error
        assert not is_infra_error("invalid api key (401)")
        assert not is_infra_error("authentication failed (403)")

    def test_model_not_found_not_infra(self):
        from silp.bench.models import is_infra_error
        assert not is_infra_error("model not found (404)")

    def test_context_length_not_infra(self):
        from silp.bench.models import is_infra_error
        assert not is_infra_error("context length exceeded")
        assert not is_infra_error("token limit exceeded")

    def test_content_filter_not_infra(self):
        from silp.bench.models import is_infra_error
        assert not is_infra_error("content filter triggered")
        assert not is_infra_error("safety block")

    def test_random_error_not_infra(self):
        from silp.bench.models import is_infra_error
        assert not is_infra_error("some random semantic error")


# ═══════════════════════════════════════════════════════════════════════
# 2. ModelResponse.retries Field
# ═══════════════════════════════════════════════════════════════════════


class TestModelResponseRetries:
    """Test that ModelResponse has a retries field with default 0."""

    def test_retries_default_zero(self):
        from silp.bench.models import ModelResponse
        resp = ModelResponse(text="hello", model="test", backend="api")
        assert resp.retries == 0

    def test_retries_set_explicitly(self):
        from silp.bench.models import ModelResponse
        resp = ModelResponse(text="hello", model="test", backend="api", retries=2)
        assert resp.retries == 2


class TestRetryMechanism:
    """Test the retry loop in ModelBackend.generate()."""

    def test_retry_on_timeout(self):
        """A backend that times out once then succeeds should return retries=1."""
        from silp.bench.models import ModelBackend, ModelResponse, GenerationConfig

        class FakeBackend(ModelBackend):
            name = "fake"
            backend_type = "api"
            max_retries = 2
            retry_backoff_base = 0.01  # fast for testing

            def __init__(self):
                self._call_count = 0

            def _generate_once(self, prompt, config):
                self._call_count += 1
                if self._call_count == 1:
                    return ModelResponse(
                        text="", model=self.name, backend="api",
                        error="Connection timeout after 30s",
                    )
                return ModelResponse(
                    text="success", model=self.name, backend="api", elapsed=0.1,
                )

        backend = FakeBackend()
        resp = backend.generate("test prompt")
        assert resp.error is None
        assert resp.text == "success"
        assert resp.retries == 1

    def test_no_retry_on_auth_error(self):
        """Auth errors should NOT be retried."""
        from silp.bench.models import ModelBackend, ModelResponse, GenerationConfig

        class FakeBackend(ModelBackend):
            name = "fake"
            backend_type = "api"
            max_retries = 2

            def __init__(self):
                self._call_count = 0

            def _generate_once(self, prompt, config):
                self._call_count += 1
                return ModelResponse(
                    text="", model=self.name, backend="api",
                    error="invalid api key (401)",
                )

        backend = FakeBackend()
        resp = backend.generate("test")
        assert resp.error is not None
        assert "401" in resp.error
        assert resp.retries == 0
        assert backend._call_count == 1  # no retry

    def test_retry_exhaustion(self):
        """All retries exhausted returns last error."""
        from silp.bench.models import ModelBackend, ModelResponse

        class FakeBackend(ModelBackend):
            name = "fake"
            backend_type = "api"
            max_retries = 1
            retry_backoff_base = 0.01

            def _generate_once(self, prompt, config):
                return ModelResponse(
                    text="", model=self.name, backend="api",
                    error="Connection timeout",
                )

        backend = FakeBackend()
        resp = backend.generate("test")
        assert resp.error is not None
        assert resp.retries == 1  # 1 retry attempt


# ═══════════════════════════════════════════════════════════════════════
# 3. NLInJSONFrontend Tests
# ═══════════════════════════════════════════════════════════════════════


class TestNLInJSONFrontend:
    """Test the natural-language-in-JSON control baseline."""

    def test_compile_produces_json(self):
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = _make_ir()
        ir = validate_ir(data, enforce_whitelist=False).ir
        fe = get_frontend("nl_json")
        result = fe.compile(ir)

        parsed = json.loads(result)
        assert "task_description" in parsed
        assert isinstance(parsed["task_description"], str)

    def test_compile_content_matches_natural(self):
        """The prose inside JSON must match NaturalFrontend output."""
        from silp.frontend import get_frontend
        from silp.ir import validate as validate_ir

        data = _make_ir()
        ir = validate_ir(data, enforce_whitelist=False).ir

        nl_fe = get_frontend("natural")
        nlj_fe = get_frontend("nl_json")

        nl_output = nl_fe.compile(ir)
        nlj_output = nlj_fe.compile(ir)
        parsed = json.loads(nlj_output)

        assert parsed["task_description"] == nl_output

    def test_decode_raises_not_implemented(self):
        from silp.frontend import get_frontend

        fe = get_frontend("nl_json")
        with pytest.raises(NotImplementedError):
            fe.decode('{"task_description": "cancel flight."}')

    def test_name_is_nl_json(self):
        from silp.frontend import get_frontend
        fe = get_frontend("nl_json")
        assert fe.name == "nl_json"


# ═══════════════════════════════════════════════════════════════════════
# 4. LLMLingua2Frontend Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLLMLingua2Frontend:
    """Test the LLMLingua-2 compression baseline."""

    def test_name_is_llmlingua2(self):
        from silp.frontend import get_frontend
        fe = get_frontend("llmlingua2")
        assert fe.name == "llmlingua2"

    def test_decode_raises_not_implemented(self):
        from silp.frontend import get_frontend

        fe = get_frontend("llmlingua2")
        with pytest.raises(NotImplementedError):
            fe.decode("compressed text")

    def test_import_does_not_crash(self):
        """Importing the module should not trigger model loading."""
        from silp.frontend.llmlingua import LLMLingua2Frontend
        # Just creating an instance should be fine (lazy load)
        fe = LLMLingua2Frontend()
        assert fe._compressor is None  # not loaded yet


# ═══════════════════════════════════════════════════════════════════════
# 5. Frontend Registry Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFrontendRegistry:
    """Test that all Phase 2 frontends are registered."""

    def test_all_frontends_registered(self):
        from silp.frontend import list_frontends
        frontends = list_frontends()
        assert "code" in frontends
        assert "json" in frontends
        assert "natural" in frontends
        assert "nl_json" in frontends
        assert "llmlingua2" in frontends

    def test_at_least_5_frontends(self):
        from silp.frontend import list_frontends
        assert len(list_frontends()) >= 5


# ═══════════════════════════════════════════════════════════════════════
# 6. Phase 2 Matrix Script Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPhase2MatrixScript:
    """Test the Phase 2 matrix runner script."""

    def test_dry_run_compiles_all_frontends(self):
        """Dry run should compile all 4 non-llmlingua2 frontends without error."""
        from run_phase2_matrix import run_matrix
        # dry_run=True should not call any models
        run_matrix(
            models=["deepseek-v3.2"],
            frontends=["code", "json", "natural", "nl_json"],
            dry_run=True,
        )

    def test_spearman_rho_perfect_correlation(self):
        from run_phase2_matrix import _spearman_rho
        # Perfect positive correlation
        assert _spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == 1.0

    def test_spearman_rho_perfect_negative(self):
        from run_phase2_matrix import _spearman_rho
        assert _spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == -1.0

    def test_spearman_rho_no_correlation(self):
        from run_phase2_matrix import _spearman_rho
        # For [1,2,3,4,5] vs [3,1,4,2,5], rho should be between -1 and 1
        rho = _spearman_rho([1, 2, 3, 4, 5], [3, 1, 4, 2, 5])
        assert -1 <= rho <= 1

    def test_spearman_rho_short_list(self):
        from run_phase2_matrix import _spearman_rho
        assert _spearman_rho([1], [2]) == 0.0

    def test_rank_with_ties(self):
        from run_phase2_matrix import _rank
        # [10, 20, 20, 30] → ranks [4, 2.5, 2.5, 1] (1=highest)
        ranks = _rank([10, 20, 20, 30])
        assert ranks[0] == 4.0  # 10 is lowest
        assert ranks[3] == 1.0  # 30 is highest
        assert ranks[1] == ranks[2]  # ties get same rank


# ═══════════════════════════════════════════════════════════════════════
# 7. Probe Output Tests
# ═══════════════════════════════════════════════════════════════════════


class TestProbeOutputs:
    """Test that probe output files exist and have data."""

    def test_zerowidth_probe_exists(self):
        path = PHASE2_PROCESSED / "probe_zerowidth.csv"
        assert path.exists(), f"Missing {path}"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 1  # header + data

    def test_trigger_head_probe_exists(self):
        path = PHASE2_PROCESSED / "probe_trigger_head.csv"
        assert path.exists(), f"Missing {path}"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 1

    def test_variance_threshold_probe_exists(self):
        path = PHASE2_PROCESSED / "probe_variance_threshold.csv"
        assert path.exists(), f"Missing {path}"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 1

    def test_zerowidth_has_delta_column(self):
        import csv
        path = PHASE2_PROCESSED / "probe_zerowidth.csv"
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "delta" in row
            assert "survived" in row

    def test_trigger_head_has_roundtrip_column(self):
        import csv
        path = PHASE2_PROCESSED / "probe_trigger_head.csv"
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "roundtrip_ok" in row
            assert "token_count" in row

    def test_variance_has_range_column(self):
        import csv
        path = PHASE2_PROCESSED / "probe_variance_threshold.csv"
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "range" in row
            assert "exceeds_threshold" in row


# ═══════════════════════════════════════════════════════════════════════
# 8. Smoke Test Runner — Retry Recording
# ═══════════════════════════════════════════════════════════════════════


class TestSmokeTestRetryRecording:
    """Verify that run_smoke_test.py records the retries field."""

    def test_smoke_test_has_retries_in_result_dict(self):
        """The smoke test result dict should include 'retries' key."""
        # Read the source and check for retries field
        source = (ROOT / "scripts" / "run_smoke_test.py").read_text("utf-8")
        assert '"retries"' in source or "'retries'" in source
