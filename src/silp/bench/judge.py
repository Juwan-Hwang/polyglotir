"""LLM-as-judge — semantic judgment (not string matching).

The judge receives the original IR, the frontend encoding shown to the model,
and the model's response. It then asks a judge LLM to determine whether the
response correctly captures the semantic intent.

Design principles (from spec §4 Phase 0):
- Judgment is **semantic**, not string-exact-match.
- The judge prompt is **fixed** — no per-case tuning.
- Judge results are recorded for audit.

Two judge modes:
1. ``llm`` — use an API model (default: gpt-4o-mini) as the judge.
2. ``rule`` — lightweight rule-based heuristic for quick local testing
   (checks for key entities/actions in the response).  This is NOT a
   replacement for the LLM judge; it's a fast pre-filter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ..ir.schema import SilpIR
from .models import GenerationConfig, ModelBackend, get_model


# ── Result ────────────────────────────────────────────────────────────


@dataclass
class JudgeResult:
    """Outcome of a single judgment."""

    verdict: str  # "pass" or "fail"
    reason: str
    judge: str  # judge model name
    details: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


# ── Judge prompt ──────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict semantic evaluator. Your job is to determine whether a model's response correctly captures the intent of a given instruction.

You will receive:
1. The original intent (in JSON IR format)
2. The encoded instruction shown to the model
3. The model's response

Evaluate whether the model's response demonstrates correct understanding of ALL of the following (if present in the IR):
- The main action (e.g., cancel, start, translate)
- All entities/arguments (e.g., what to cancel, what to translate)
- All constraints/conditions (e.g., "if not raining", "if budget <= 500")
- All alternative/else-branch actions
- The correct ordering if sequence is specified
- Negation logic (e.g., "if NOT rain" must not be confused with "if rain")

Respond in EXACTLY this JSON format:
{"verdict": "pass" or "fail", "reason": "one-sentence explanation"}

A response passes if it correctly captures the FULL semantic intent, even if the wording differs. A response fails if it misses any entity, reverses any condition, or omits any required action."""

JUDGE_PROMPT_TEMPLATE = """## Original Intent (IR)
{ir_json}

## Encoded Instruction (shown to model)
{encoded}

## Model Response
{response}

Evaluate: does the model's response correctly capture the full semantic intent?"""


# ── LLM Judge ─────────────────────────────────────────────────────────


class LLMJudge:
    """LLM-based semantic judge.

    Uses a separate model (default: gpt-4o-mini) to evaluate whether
    a model response correctly captures the IR's intent.
    """

    def __init__(
        self,
        judge_model_name: str = "gpt-4o-mini",
        model: ModelBackend | None = None,
    ) -> None:
        self.judge_model_name = judge_model_name
        self._model = model or get_model(judge_model_name)

    def judge(
        self,
        ir: SilpIR,
        encoded: str,
        model_response: str,
    ) -> JudgeResult:
        """Judge whether *model_response* correctly captures *ir*."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            ir_json=ir.to_compact_json(),
            encoded=encoded,
            response=model_response,
        )

        full_prompt = f"{JUDGE_SYSTEM}\n\n{prompt}"
        response = self._model.generate(
            full_prompt,
            GenerationConfig(max_new_tokens=128, temperature=0.0),
        )

        if response.error:
            return JudgeResult(
                verdict="fail",
                reason=f"Judge error: {response.error}",
                judge=self.judge_model_name,
            )

        return _parse_judge_response(response.text, self.judge_model_name)


def _parse_judge_response(text: str, judge_name: str) -> JudgeResult:
    """Parse the judge LLM's JSON response."""
    # Try to extract JSON from the response
    # The judge might wrap it in ```json ... ``` or add extra text
    json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            verdict = data.get("verdict", "fail").lower().strip()
            reason = data.get("reason", "no reason provided")
            if verdict not in ("pass", "fail"):
                verdict = "fail"
            return JudgeResult(
                verdict=verdict,
                reason=reason,
                judge=judge_name,
                details={"raw": text},
            )
        except json.JSONDecodeError:
            pass

    # Fallback: check for "pass" or "fail" in the text
    text_lower = text.lower()
    if '"pass"' in text_lower or "verdict: pass" in text_lower:
        return JudgeResult(
            verdict="pass",
            reason=text[:200],
            judge=judge_name,
        )
    return JudgeResult(
        verdict="fail",
        reason=f"Could not parse judge response: {text[:200]}",
        judge=judge_name,
    )


# ── Rule-based judge (fast pre-filter) ────────────────────────────────


class RuleJudge:
    """Lightweight rule-based judge for quick local testing.

    Checks that key entities and actions from the IR appear in the response.
    This is NOT a substitute for the LLM judge — it's a fast pre-filter
    that catches obvious failures.

    Rules:
    1. Main action verb must appear (e.g., "cancel" for !CANCEL)
    2. All entity values must appear (or close variants)
    3. Negation must be preserved (if IR has !rain, response must not
       imply the opposite)
    """

    def judge(
        self,
        ir: SilpIR,
        encoded: str,
        model_response: str,
    ) -> JudgeResult:
        response_lower = model_response.lower()
        details: dict[str, object] = {"checks": []}

        # 1. Check main action verb (and all its synonyms)
        verb = ir.intent[1:].lower()
        candidates = [verb] + _VERB_SYNONYMS.get(verb, [])
        verb_found = any(c in response_lower for c in candidates)
        details["checks"].append({
            "check": "main_verb",
            "expected": verb,
            "found": verb_found,
        })

        # 2. Check entity values
        missing_entities = []
        for e in ir.entities:
            # Normalize: lowercase, replace _ with space
            val = e.value.lower().replace("_", " ")
            found = val in response_lower or e.value.lower() in response_lower
            if not found:
                missing_entities.append(e.value)
        details["checks"].append({
            "check": "entities",
            "missing": missing_entities,
        })

        # 3. Check negation
        negation_ok = True
        for c in ir.constraints:
            if c.type.startswith("!"):
                negated = c.type[1:].lower()
                # If the response says the negated thing IS happening, that's wrong
                # e.g., if IR says !rain, response saying "it will rain" is wrong
                # This is a crude heuristic — the LLM judge handles this properly
                if f"{negated}" in response_lower and "not" not in response_lower:
                    negation_ok = False
        details["checks"].append({
            "check": "negation",
            "ok": negation_ok,
        })

        # Aggregate
        all_ok = verb_found and not missing_entities and negation_ok
        if all_ok:
            return JudgeResult(
                verdict="pass",
                reason="Rule-based: all key entities and verbs found",
                judge="rule",
                details=details,
            )
        reasons = []
        if not verb_found:
            reasons.append(f"main verb '{verb}' not found")
        if missing_entities:
            reasons.append(f"missing entities: {missing_entities}")
        if not negation_ok:
            reasons.append("negation may be reversed")
        return JudgeResult(
            verdict="fail",
            reason="; ".join(reasons),
            judge="rule",
            details=details,
        )


# ── Synonyms (minimal, for rule judge) ────────────────────────────────

_VERB_SYNONYMS: dict[str, list[str]] = {
    "cancel": ["cancell", "void", "abort"],
    "email": ["notify", "send", "message", "contact"],
    "start": ["begin", "launch", "initiate"],
    "fetch": ["get", "retrieve", "obtain"],
    "process": ["handle", "compute", "analyze"],
    "translate": ["convert", "transform"],
    "book": ["reserve", "schedule"],
    "route": ["direct", "assign", "transfer"],
    "search": ["find", "lookup", "query"],
    "update": ["modify", "change", "edit"],
    "escalate": ["raise", "promote", "advance"],
    "suggest": ["recommend", "propose"],
}


def _verb_synonym(verb: str) -> str:
    """Return the first synonym for *verb*, or the verb itself."""
    synonyms = _VERB_SYNONYMS.get(verb, [])
    return synonyms[0] if synonyms else verb


# ── Convenience ───────────────────────────────────────────────────────


def get_judge(
    mode: str = "rule",
    judge_model: str = "gpt-4o-mini",
) -> LLMJudge | RuleJudge:
    """Get a judge instance.

    Args:
        mode: "rule" for fast rule-based, "llm" for LLM-based.
        judge_model: Model name for LLM judge (ignored if mode="rule").
    """
    if mode == "llm":
        return LLMJudge(judge_model_name=judge_model)
    return RuleJudge()
