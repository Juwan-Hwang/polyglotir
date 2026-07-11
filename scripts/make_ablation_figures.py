"""Generate ablation figures from raw results."""
import json, collections, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "phase3_ablation"
FIGURES = ROOT / "data" / "figures"
OUTPUT = ROOT / "data" / "processed" / "phase3_ablation"

MODELS = ["deepseek-v3-2", "glm-5-2", "kimi-k2-6"]
MODEL_KEYS = {"deepseek-v3-2": "deepseek-v3.2", "glm-5-2": "glm-5.2", "kimi-k2-6": "kimi-k2.6"}
CONDITIONS = ["original", "shuffled", "skeleton"]
CASE_TYPES_ORDER = [
    ("case10", "multi_turn"), ("case1", "multi_constraint"), ("case2", "negation"),
    ("case3", "detail"), ("case5", "tool_branch"), ("case6", "nested_constraint"),
    ("case7", "parallel_action"), ("case8", "conditional_branch"), ("case9", "tool_call"),
]

def get_case_type(case_id):
    for prefix, ctype in CASE_TYPES_ORDER:
        if case_id.startswith(prefix):
            return ctype
    return "unknown"

# Load
all_results = []
for slug in MODELS:
    for line in (RAW / slug / "results.jsonl").read_text("utf-8").strip().split("\n"):
        all_results.append(json.loads(line))

# Write matrix CSV
OUTPUT.mkdir(parents=True, exist_ok=True)
import csv
with open(OUTPUT / "ablation_matrix.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition"] + [MODEL_KEYS[m] for m in MODELS] + ["overall"])
    for c in CONDITIONS:
        row = [c]
        for slug in MODELS:
            mk = MODEL_KEYS[slug]
            s = [r for r in all_results if r["model"] == mk and r["condition"] == c]
            p = sum(1 for r in s if r["first_pass"])
            row.append(f"{p}/{len(s)} ({p/len(s)*100:.1f}%)")
        s = [r for r in all_results if r["condition"] == c]
        p = sum(1 for r in s if r["first_pass"])
        row.append(f"{p}/{len(s)} ({p/len(s)*100:.1f}%)")
        w.writerow(row)

# Per-case-type CSV
with open(OUTPUT / "ablation_by_type.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["case_type", "original", "shuffled", "skeleton"])
    by_tc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in all_results:
        ct = get_case_type(r["case_id"])
        by_tc[ct][r["condition"]][1] += 1
        if r["first_pass"]:
            by_tc[ct][r["condition"]][0] += 1
    for ct in sorted(by_tc):
        row = [ct]
        for c in CONDITIONS:
            p, t = by_tc[ct][c]
            row.append(f"{p}/{t} ({p/t*100:.0f}%)" if t else "N/A")
        w.writerow(row)

print(f"CSV: {OUTPUT / 'ablation_matrix.csv'}")
print(f"CSV: {OUTPUT / 'ablation_by_type.csv'}")

# Figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES.mkdir(parents=True, exist_ok=True)

    # Fig 1: Overall bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    rates = []
    for c in CONDITIONS:
        s = [r for r in all_results if r["condition"] == c]
        p = sum(1 for r in s if r["first_pass"])
        rates.append(p / len(s) * 100)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(CONDITIONS, rates, color=colors, width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Ablation: Syntax vs Vocabulary Priors (n=243)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_overall.png", dpi=150)
    print(f"Fig: {FIGURES / 'ablation_overall.png'}")

    # Fig 2: Per-case-type grouped bar
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    types = sorted(by_tc.keys())
    x = np.arange(len(types))
    width = 0.25
    for i, c in enumerate(CONDITIONS):
        vals = []
        for ct in types:
            p, t = by_tc[ct][c]
            vals.append(p/t*100 if t else 0)
        ax2.bar(x + i*width, vals, width, label=c, color=colors[i])
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_title("Ablation by Case Type", fontsize=14)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(types, rotation=30, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 115)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "ablation_by_type.png", dpi=150)
    print(f"Fig: {FIGURES / 'ablation_by_type.png'}")

except ImportError:
    print("matplotlib not available")
