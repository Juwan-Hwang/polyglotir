"""Patch the two 401 error results by re-running just those two cells."""
import sys, json, time
sys.path.insert(0, "src")
from pathlib import Path
from datetime import datetime, timezone
from silp.bench.models import GenerationConfig, get_model, load_env
from silp.bench.judge import get_judge
from silp.frontend import get_frontend as get_fe
from silp.ir import validate as validate_ir

load_env()

DECODE_PROMPT = """Decode the following SILP payload and explain what action(s) should be taken. Describe the full intent including all conditions, entities, and alternatives.

SILP payload:
{encoded}

Explain the semantic intent:"""

targets = [
    ("longcat-2.0", "case8c_conditional_branch", "nl_json"),
    ("minimax-m2.7", "case3b_detail", "llmlingua2"),
]

judge = get_judge("dual", "glm-5.2")

for model_name, case_id, fe_name in targets:
    model_slug = model_name.replace(".", "-").replace("/", "-")
    jsonl_path = Path(f"data/raw/phase2/{model_slug}/results.jsonl")

    ir_data = json.loads(Path(f"examples/{case_id}.json").read_text(encoding="utf-8"))
    ir = validate_ir(ir_data).ir
    fe = get_fe(fe_name)
    encoded = fe.compile(ir)
    prompt = DECODE_PROMPT.format(encoded=encoded)
    model = get_model(model_name)

    # Infinite retry until we get a real response
    outer = 0
    while True:
        resp = model.generate(prompt, GenerationConfig(max_new_tokens=256, temperature=0.0, timeout=30.0))
        if not resp.error:
            break
        outer += 1
        print(f"  retry #{outer} for {case_id}|{fe_name}|{model_name}: {resp.error[:60]}")
        time.sleep(min(30, 5 * outer))

    jr = judge.judge(ir, encoded, resp.text)
    details = jr.details
    result = {
        "case_id": case_id, "frontend": fe_name, "model": model_name,
        "encoded": encoded, "model_response": resp.text,
        "judge_verdict": jr.verdict, "judge_reason": jr.reason, "judge": jr.judge,
        "rule_verdict": details.get("rule_verdict"), "rule_reason": details.get("rule_reason"),
        "llm_verdict": details.get("llm_verdict"), "llm_reason": details.get("llm_reason"),
        "elapsed": resp.elapsed, "retries": resp.retries + outer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "first_pass": jr.passed,
    }

    # Replace the error line in-place (NOT overwrite the whole file)
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    new_lines = []
    for line in lines:
        r = json.loads(line)
        if r["case_id"] == case_id and r["frontend"] == fe_name and r["model"] == model_name:
            new_lines.append(json.dumps(result, ensure_ascii=False))
        else:
            new_lines.append(line)
    jsonl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    status = "PASS" if jr.passed else "FAIL"
    print(f"  {case_id}|{fe_name}|{model_name} -> {status} ({resp.elapsed:.1f}s, {outer} retries)")
