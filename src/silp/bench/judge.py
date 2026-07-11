"""LLM-as-judge — semantic judgment (not string matching).

The judge receives the original IR, the frontend encoding shown to the model,
and the model's response. It then asks a judge LLM to determine whether the
response correctly captures the semantic intent.

Design principles (from spec §4 Phase 0):
- Judgment is **semantic**, not string-exact-match.
- The judge prompt is **fixed** — no per-case tuning.
- Judge results are recorded for audit.

Three judge modes:
1. ``llm``  — use an API model as the judge (default: glm-5.2).
2. ``rule`` — lightweight rule-based heuristic for quick local testing
   (checks for key entities/actions in the response, with Chinese synonym
   support).  This is NOT a replacement for the LLM judge.
3. ``dual`` — run both rule and LLM judges, record both verdicts.
   The LLM verdict is primary; the rule verdict is recorded for analysis
   (e.g., measuring the false-negative rate of rule-based judges in
   cross-lingual scenarios — itself a finding for the paper).
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
    judge: str  # judge model name or "rule"
    details: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


# ── Judge prompt (bilingual-aware) ────────────────────────────────────

JUDGE_SYSTEM = """You are a strict semantic evaluator for the SILP (Semantic Interlingua Layer Protocol) project.

Your task: determine whether a model's response correctly captures the FULL semantic intent of a SILP instruction. The response may be in English, Chinese, or any other language — judge by SEMANTIC equivalence, not string matching.

You will receive:
1. Original Intent (JSON IR format) — the canonical semantic representation
2. Encoded Instruction — what was shown to the model
3. Model Response — the model's interpretation

Evaluation criteria (ALL must be correct for a "pass"):
- Main action: The response must correctly identify the primary action (e.g., cancel, start, translate, update). Paraphrasing in any language is acceptable.
- All entities/arguments: Every entity value and its role must be correctly identified. Paraphrasing is acceptable if the semantic meaning is preserved (e.g., "order_42" can be described as "订单42" or "order number 42").
- All constraints/conditions: Every condition must be correctly captured, including time bounds, subjects, and operators (e.g., <=, !=, >=).
- Negation logic: Negated conditions (e.g., "if NOT rain") must NOT be reversed. A reversed negation is an automatic fail. This is the most critical check.
- Alternative/else-branch: All fallback actions must be correctly identified with their targets and locations.
- Ordering: If meta.seq specifies a sequence, the order must be preserved.

A response FAILS if:
- It misses or misunderstands any entity, constraint, or action
- It reverses any negation condition (e.g., saying "if rain" when the IR says "if NOT rain")
- It omits any required action or alternative
- It fabricates entities or conditions not present in the IR

Respond in EXACTLY this JSON format (no markdown fences, no extra text):
{"verdict": "pass" or "fail", "reason": "one-sentence explanation", "missing": ["list of any missing or incorrect elements, empty array if pass"]}"""

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

    Uses a separate model (default: glm-5.2) to evaluate whether
    a model response correctly captures the IR's intent.
    """

    def __init__(
        self,
        judge_model_name: str = "glm-5.2",
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
            GenerationConfig(max_new_tokens=256, temperature=0.0),
        )

        if response.error:
            return JudgeResult(
                verdict="fail",
                reason=f"Judge error: {response.error}",
                judge=self.judge_model_name,
            )

        return _parse_judge_response(response.text, self.judge_model_name)


def _parse_judge_response(text: str, judge_name: str) -> JudgeResult:
    """Parse the judge LLM's JSON response.

    Handles:
    - Plain JSON
    - JSON wrapped in markdown code fences
    - JSON with extra text before/after
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        return _build_judge_result(data, judge_name, text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = cleaned[json_start : json_end + 1]
        try:
            data = json.loads(json_str)
            return _build_judge_result(data, judge_name, text)
        except json.JSONDecodeError:
            pass

    # Fallback: check for "pass" or "fail" in text
    text_lower = cleaned.lower()
    if '"pass"' in text_lower or "verdict: pass" in text_lower:
        return JudgeResult(
            verdict="pass",
            reason=cleaned[:200],
            judge=judge_name,
            details={"raw": text},
        )
    return JudgeResult(
        verdict="fail",
        reason=f"Could not parse judge response: {cleaned[:200]}",
        judge=judge_name,
        details={"raw": text},
    )


def _build_judge_result(data: dict, judge_name: str, raw: str) -> JudgeResult:
    """Build a JudgeResult from parsed JSON data."""
    verdict = str(data.get("verdict", "fail")).lower().strip()
    if verdict not in ("pass", "fail"):
        verdict = "fail"
    reason = str(data.get("reason", "no reason provided"))
    missing = data.get("missing", [])
    return JudgeResult(
        verdict=verdict,
        reason=reason,
        judge=judge_name,
        details={"missing": missing, "raw": raw},
    )


# ── Rule-based judge (fast pre-filter, Chinese-aware) ─────────────────


# ── Bilingual synonym tables ─────────────────────────────────────────
#
# Phase 0.5 analysis revealed four systematic false-negative patterns in
# the rule judge:
#
#   1. Chinese translation  — model says "北京" but rule looks for "beijing"
#   2. Boolean sentinels    — constraint value "true" is meaningless to search
#   3. Compound decomposition — "fr_rev_bold" is explained as fr+rev+bold
#   4. Unit normalisation   — "30s" appears as "30秒" / "30 seconds"
#
# The tables and logic below address all four.  False positives (fabrication,
# semantic misunderstanding) are inherently beyond rule-based judging and
# remain the responsibility of the LLM judge.

# Verb synonyms: English + Chinese.
_VERB_SYNONYMS: dict[str, list[str]] = {
    "cancel": ["取消", "撤销", "废除", "cancell", "void", "abort"],
    "start": ["开始", "启动", "发起", "begin", "launch", "initiate"],
    "translate": ["翻译", "转换", "转化", "convert", "transform"],
    "update": ["更新", "修改", "编辑", "change", "modify", "edit"],
    "email": ["邮件", "通知", "发送", "信件", "notify", "send", "message", "contact"],
    "fetch": ["获取", "提取", "取得", "get", "retrieve", "obtain"],
    "process": ["处理", "计算", "分析", "handle", "compute", "analyze"],
    "book": ["预订", "预约", "订购", "reserve", "schedule"],
    "route": ["路由", "转派", "分配", "direct", "assign", "transfer"],
    "search": ["搜索", "查找", "检索", "find", "lookup", "query"],
    "switch": ["切换", "转换", "切换工具", "switch_tool"],
    "switch_tool": ["切换", "转换工具", "切换工具", "switch"],
    "escalate": ["升级", "上报", "提升", "raise", "promote", "advance"],
    "suggest": ["建议", "推荐", "提议", "recommend", "propose"],
}

# Bilingual entity-value map: English → Chinese equivalents.
# Keys are lowercased.  Values are lists of substrings to search for in the
# (already lowercased) response.  Order: most specific first.
_ENTITY_CN_MAP: dict[str, list[str]] = {
    # ── Locations ──
    "beijing": ["北京"],
    "shanghai": ["上海"],
    "downtown": ["市中心", "市区"],
    # ── Travel ──
    "flight": ["航班", "机票", "飞机"],
    "hotel": ["酒店", "旅馆"],
    # ── Activities ──
    "hike": ["徒步", "远足", "登山"],
    "cards": ["纸牌", "打牌", "卡牌"],
    "indoor": ["室内"],
    "indoor_activity": ["室内活动"],
    # ── Food / search ──
    "restaurants": ["餐厅"],
    "italian": ["意大利"],
    "delivery_options": ["外卖", "配送"],
    "open_now": ["营业中", "正在营业", "营业"],
    # ── Business / ticketing ──
    "billing": ["账单", "计费"],
    "manager": ["经理", "主管"],
    "tech_support": ["技术支持"],
    "ticket": ["工单"],
    # ── Data objects ──
    "data": ["数据"],
    "report": ["报告"],
    "result": ["结果"],
    # ── Weather ──
    "weather": ["天气"],
    "rain": ["雨", "下雨"],
    # ── Booking constraints ──
    "budget": ["预算"],
    "rating": ["评分"],
    "urgency": ["紧急", "优先级"],
    "high": ["高"],
    "category": ["类别", "分类"],
    "shipped": ["已发货", "发货"],
    "verified": ["已验证", "已认证", "验证通过"],
    # ── System / protocol ──
    "timeout": ["超时"],
    "auth": ["认证", "身份验证"],
    "status": ["状态"],
    "loc": ["位置", "地点"],
    "address": ["地址"],
    # ── Style values (compound — also handled by decomposition) ──
    "archaic_heavy": ["重度古风", "重度古语", "古风重度"],
    "shakespeare_en": ["莎士比亚", "莎翁"],
    "fr_rev_bold": ["法语革命", "法文革命"],
}

# Boolean sentinel values — meaningless to search for in a response.
# These appear as constraint ``value`` when the ``type`` encodes the real
# semantics (e.g. ``{"type": "!rain", "value": "true"}``).
_SENTINEL_VALUES = frozenset({"true", "false", "null", "none", ""})

# Negation markers: English + Chinese.
_NEGATION_MARKERS = [
    # English
    "not", "no ", "without", "isn't", "aren't", "don't", "doesn't",
    "won't", "can't", "cannot",
    # Chinese
    "不", "没", "无", "非", "否", "未",
]


class RuleJudge:
    """Lightweight rule-based judge for quick local testing.

    Checks that key entities and actions from the IR appear in the response.
    Supports both English and Chinese responses via synonym tables.

    This is NOT a substitute for the LLM judge — it's a fast pre-filter
    that catches obvious failures. Its false-negative rate in cross-lingual
    scenarios is itself a measurable metric for the paper.
    """

    def judge(
        self,
        ir: SilpIR,
        encoded: str,
        model_response: str,
    ) -> JudgeResult:
        response_lower = model_response.lower()
        details: dict[str, object] = {"checks": []}

        # 1. Check main action verb (English + Chinese synonyms)
        verb = ir.intent[1:].lower()
        candidates = [verb] + _VERB_SYNONYMS.get(verb, [])
        verb_found = any(c.lower() in response_lower for c in candidates)
        details["checks"].append({
            "check": "main_verb",
            "expected": verb,
            "found": verb_found,
        })

        # 2. Check entity values (with normalization)
        missing_entities = []
        for e in ir.entities:
            if _find_value(e.value, response_lower):
                continue
            missing_entities.append(e.value)
        details["checks"].append({
            "check": "entities",
            "missing": missing_entities,
        })

        # 3. Check constraint values
        missing_constraints = []
        for c in ir.constraints:
            if _find_value(c.value, response_lower):
                continue
            missing_constraints.append(c.value)
        if missing_constraints:
            details["checks"].append({
                "check": "constraint_values",
                "missing": missing_constraints,
            })

        # 4. Check alternative targets
        missing_alts = []
        for alt in ir.alternatives:
            if alt.target and not _find_value(alt.target, response_lower):
                missing_alts.append(alt.target)
        if missing_alts:
            details["checks"].append({
                "check": "alternative_targets",
                "missing": missing_alts,
            })

        # 5. Check negation
        negation_ok = True
        for c in ir.constraints:
            if c.type.startswith("!"):
                negated = c.type[1:].lower()
                # If the negated word appears, check for negation markers
                if negated in response_lower:
                    has_negation = any(m in response_lower for m in _NEGATION_MARKERS)
                    if not has_negation:
                        negation_ok = False
        details["checks"].append({
            "check": "negation",
            "ok": negation_ok,
        })

        # Aggregate
        all_missing = missing_entities + missing_constraints + missing_alts
        all_ok = verb_found and not all_missing and negation_ok
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
        if missing_constraints:
            reasons.append(f"missing constraints: {missing_constraints}")
        if missing_alts:
            reasons.append(f"missing alternatives: {missing_alts}")
        if not negation_ok:
            reasons.append("negation may be reversed")
        return JudgeResult(
            verdict="fail",
            reason="; ".join(reasons),
            judge="rule",
            details=details,
        )


def _find_value(value: str, response_lower: str) -> bool:
    """Check if *value* appears in *response_lower*, with normalization.

    Handles (in priority order):
    1. Boolean sentinels (``"true"``/``"false"``) → always match
    2. Exact match (lowercase)
    3. Underscore variants (``order_42`` → ``order 42`` / ``order42``)
    4. Chinese translation (``beijing`` → ``北京``)
    5. Compound decomposition (``fr_rev_bold`` → all of fr, rev, bold)
    6. Time-unit normalisation (``30s`` → ``30秒`` / ``30 seconds``)
    """
    val_lower = value.lower()

    # 1. Boolean sentinels — meaningless to search for
    if val_lower in _SENTINEL_VALUES:
        return True

    # 2–4. Build and check variant list
    variants = [
        val_lower,
        val_lower.replace("_", " "),
        val_lower.replace("_", ""),
    ]
    if val_lower in _ENTITY_CN_MAP:
        variants.extend(_ENTITY_CN_MAP[val_lower])
    if any(v in response_lower for v in variants):
        return True

    # 5. Compound decomposition: check each underscore-separated part
    if "_" in val_lower:
        parts = val_lower.split("_")
        part_lookups = []
        for part in parts:
            pv = [part]
            if part in _ENTITY_CN_MAP:
                pv.extend(_ENTITY_CN_MAP[part])
            part_lookups.append(pv)
        if all(any(v in response_lower for v in pvs) for pvs in part_lookups):
            return True

    # 6. Time-unit normalisation: ``30s`` → ``30``, ``30秒``, ``30 seconds``
    for suffix in ("ms", "min", "sec", "s"):
        if val_lower.endswith(suffix) and len(val_lower) > len(suffix):
            numeric = val_lower[: -len(suffix)]
            if numeric and any(c.isdigit() for c in numeric):
                unit_variants = [
                    numeric,
                    f"{numeric}秒",
                    f"{numeric} second",
                    f"{numeric} seconds",
                    f"{numeric}-second",
                ]
                if any(v in response_lower for v in unit_variants):
                    return True
            break  # only try the longest matching suffix

    return False


# ── Dual Judge (rule + LLM) ───────────────────────────────────────────


class DualJudge:
    """Runs both RuleJudge and LLMJudge.

    The LLM verdict is primary (used for pass-rate calculation).
    The rule verdict is recorded for analysis — e.g., measuring the
    false-negative rate of rule-based judges in cross-lingual scenarios,
    which is itself a finding for the paper.

    If the LLM judge errors (e.g., proxy down), falls back to the
    rule judge verdict with a note.
    """

    def __init__(
        self,
        judge_model_name: str = "glm-5.2",
        model: ModelBackend | None = None,
    ) -> None:
        self._rule = RuleJudge()
        self._llm = LLMJudge(judge_model_name, model)

    def judge(
        self,
        ir: SilpIR,
        encoded: str,
        model_response: str,
    ) -> JudgeResult:
        rule_result = self._rule.judge(ir, encoded, model_response)
        llm_result = self._llm.judge(ir, encoded, model_response)

        # If LLM judge errored, fall back to rule judge
        if "Judge error" in llm_result.reason:
            return JudgeResult(
                verdict=rule_result.verdict,
                reason=f"LLM judge unavailable, rule fallback: {rule_result.reason}",
                judge="dual_fallback",
                details={
                    "rule_verdict": rule_result.verdict,
                    "rule_reason": rule_result.reason,
                    "llm_verdict": "error",
                    "llm_reason": llm_result.reason,
                },
            )

        return JudgeResult(
            verdict=llm_result.verdict,
            reason=llm_result.reason,
            judge="dual",
            details={
                "rule_verdict": rule_result.verdict,
                "rule_reason": rule_result.reason,
                "llm_verdict": llm_result.verdict,
                "llm_reason": llm_result.reason,
            },
        )


# ── Convenience ───────────────────────────────────────────────────────


def get_judge(
    mode: str = "dual",
    judge_model: str = "glm-5.2",
) -> LLMJudge | RuleJudge | DualJudge:
    """Get a judge instance.

    Args:
        mode: "rule" for fast rule-based, "llm" for LLM-based,
              "dual" for both (default).
        judge_model: Model name for LLM judge (ignored if mode="rule").
    """
    if mode == "llm":
        return LLMJudge(judge_model)
    if mode == "dual":
        return DualJudge(judge_model)
    return RuleJudge()
