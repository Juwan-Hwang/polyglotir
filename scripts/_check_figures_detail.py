"""Deep analysis of each figure's code for text overlap / rendering issues."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "data", "figures")

issues = []

print("=" * 80)
print("DETAILED FIGURE CODE ANALYSIS - Text Overlap & Rendering Issues")
print("=" * 80)

# ── Figure 1: Pass-rate heatmap ──────────────────────────────────────
print("\n[Figure 1] fig1_passrate_heatmap.png (make_figures.py:fig_passrate_heatmap)")
print("  Code: figsize=(8, 4.5), 5 frontends x 5 models, cell annotations at fontsize=11")
print("  Model labels: rotation=30, ha='right'")
print("  Status: OK - standard heatmap, rotation=30 is safe for 5 short model names")

# ── Figure 2: Spearman bar chart ─────────────────────────────────────
print("\n[Figure 2] fig2_spearman_bar.png (make_figures.py:fig_spearman_scatter)")
print("  Code: figsize=(9, 5), 10 pairs, y-labels with '\\n' (3 lines per label!)")
print("  Label format: 'deepseek-v3.2\\nvs\\nglm-5.2' etc.")
print("  Y-tick fontsize=9, bar height=0.6")
print("  rho annotation: text at (rho+0.01, i), fontsize=9")
print("  xlim=(0, 1.1) -> rho text at max ~1.0+0.01=1.01, within range")
issues.append(("Figure 2", "HIGH_RISK",
    "Y-axis labels use 3-line format ('model_a\\nvs\\nmodel_b') with fontsize=9.\n"
    "  10 labels x 3 lines = 30 text lines in ~5in height.\n"
    "  Labels may overlap vertically, especially for long names like 'deepseek-v3.2'."))

# ── Figure 3: Token variance ─────────────────────────────────────────
print("\n[Figure 3] fig3_token_variance.png (make_figures.py:fig_token_variance)")
print("  Code: figsize=(7, 4.5), 5 bars, error bars, value labels at fontsize=10 bold")
print("  Value label position: y=bar_height+0.3")
print("  Title: 2 lines (with \\n)")
print("  Status: OK - only 5 bars, plenty of space")

# ── Figure 4: Frontend ranking ───────────────────────────────────────
print("\n[Figure 4] fig4_frontend_ranking.png (make_figures.py:fig_frontend_ranking)")
print("  Code: figsize=(7, 4), 5 bars sorted descending, labels at fontsize=11 bold")
print("  Value position: y=bar_height+0.5, ylim=max*1.2")
print("  Status: OK - clean bar chart, ample room")

# ── Figure 5: Case-level grid (CRITICAL) ─────────────────────────────
print("\n[Figure 5] fig5_case_level_grid.png (make_figures.py:fig_case_level_heatmap)")
print("  Code: figsize=(3.5 * n_fe, 5) where n_fe=5 => figsize=(17.5, 5)")
print("  ACTUAL SIZE: 5211x1540 pixels (aspect 3.38:1 - extremely wide!)")
print("  Subplots: 5 frontends side-by-side, sharey=True")
print("  X-axis: 5 models, rotation=45, ha='right', fontsize=8")
print("  Y-axis: 27 cases (!), fontsize=8, ONLY on first subplot")
print("  Cell annotations: symbols checkmark/x at fontsize=12")
print("  suptitle at y=1.02 (may overlap with subplot titles)")
issues.append(("Figure 5", "CRITICAL",
    "1. EXTREMELY WIDE (5211px / aspect 3.38:1) - 5 subplots crammed horizontally.\n"
    "   When scaled to paper width (~7in), each subplot gets only ~1.4in - TOO NARROW.\n"
    "2. 27 case labels on Y-axis at fontsize=8 in 5in height -> very cramped.\n"
    "3. X-labels rotation=45 with 5 model names in narrow subplots may overlap.\n"
    "4. suptitle at y=1.02 may overlap with 5 subplot titles."))

# ── Figure 6: Tradeoff curve ─────────────────────────────────────────
print("\n[Figure 6] tradeoff_curve.png (make_tradeoff_curve.py)")
print("  Code: figsize=(9, 6), scatter + annotate with offset(10,5), fontsize=11 bold")
print("  5 points annotated, Pareto frontier line")
print("  Legend: 80% threshold + Pareto frontier")
print("  Status: LIKELY OK but check if LLMLingua2 and NL+JSON annotations overlap\n"
    "   (both are in lower-left region of the plot)")

# ── Figure 7: Ablation overall ────────────────────────────────────────
print("\n[Figure 7] ablation_overall.png (make_ablation_figures.py)")
print("  Code: figsize=(7, 5), 3 bars (original/shuffled/skeleton)")
print("  Labels at fontsize=13 bold, y=height+1.5, ylim=(0,100)")
print("  Status: OK - only 3 bars, very spacious")

# ── Figure 8: Ablation by type (RISK) ────────────────────────────────
print("\n[Figure 8] ablation_by_type.png (make_ablation_figures.py)")
print("  Code: figsize=(12, 6), 9 categories x 3 conditions grouped bars")
print("  Bar width=0.25, x-labels rotation=30, ha='right'")
print("  Category labels: 9 names like 'conditional_branch', 'parallel_action' etc.")
print("  ylim=(0, 115) - extra headroom for legend/labels")
print("  Status: MODERATE RISK - 9 category labels at rotation=30 in 12in width")
issues.append(("Figure 8", "MODERATE",
    "9 category labels (e.g., 'conditional_branch', 'multi_constraint')\n"
    "  at rotation=30 in 12in figure width. Long label names may overlap\n"
    "  with neighboring labels or the legend box."))

# ── Figure 9: Entropy curve ──────────────────────────────────────────
print("\n[Figure 9] entropy_curve.png (make_entropy_analysis.py)")
print("  Code: figsize=(9, 7), 5 points scatter + annotate offset(10,5)")
print("  Stats text box at bottom-left (wheat color, 3 lines)")
print("  Optional: compression sweep line + noise sweep line with annotations")
print("  Status: CHECK if sweep data exists - additional annotations could crowd\n"
    "   the lower-right area where NL+JSON and LLMLingua2 already sit")

# ── Figure 10: Heartbeat pass rate by context ────────────────────────
print("\n[Figure 10] fig_heartbeat_passrate_by_context.png (make_heartbeat_analysis.py)")
print("  Code: figsize=(8, 5), line plot 3 models, markersize=6")
print("  Status: OK - simple line plot, no crowding issues")

# ── Figure 11: Error propagation ─────────────────────────────────────
print("\n[Figure 11] fig_heartbeat_error_propagation.png (make_heartbeat_analysis.py)")
print("  Code: figsize=(7, 5), grouped bar: 2 categories x 3 models")
print("  Bar width=0.25, annotation: '{val:.1f}%\\n(n={n})' at fontsize=8 bold")
print("  Annotation: 2-line text above each bar (percentage + sample size)")
print("  ylim=(0,100)")
issues.append(("Figure 11", "MODERATE",
    "Bar annotations use 2-LINE format ('23.0%\\n(n=536)') at fontsize=8.\n"
    "  With 3 models x 2 groups = 6 bars, the 2-line annotations between\n"
    "  adjacent model groups may overlap, especially when values are similar.\n"
    "  Also legend at upper-left + title 2-lines compete for space."))

# ── Figure 12: N-value overall ───────────────────────────────────────
print("\n[Figure 12] fig_heartbeat_n_overall.png (make_heartbeat_analysis.py)")
print("  Code: figsize=(9, 5), 6 N-values x 3 models grouped bars")
print("  X-labels: multi-line like 'N=1\\n(stateless)', 'N=\\u221e\\n(no heartbeat)'")
print("  Bar width=0.25, annotation fontsize=9 bold")
print("  X-tick fontsize=9")
issues.append(("Figure 12", "MODERATE",
    "X-axis labels use 2-LINE format ('N=1\\n(stateless)', 'N=INF\\n(no heartbeat)')\n"
    "  at fontsize=9. 6 groups x 3 bars with 2-line tick labels in 9in width.\n"
    "  The 'N=INF(no heartbeat)' label is especially long and may overlap."))

# ── Figure 13: Turn heatmap (CRITICAL) ───────────────────────────────
print("\n[Figure 13] fig_heartbeat_turn_heatmap.png (make_heartbeat_analysis.py)")
print("  Code: figsize=(5 * n_models, 5) where n_models=3 => figsize=(15, 5)")
print("  ACTUAL SIZE: 3651x1554 pixels (aspect 2.35:1 - very wide)")
print("  3 heatmaps side-by-side: 6 N-values (rows) x 15 turns (cols)")
print("  Cell annotations: fontsize=8 bold, values 0-100")
print("  Y-labels: 6 N-labels (some with spaces replacing \\n), fontsize=9")
print("  X-labels: turn numbers 0-14, fontsize=9")
issues.append(("Figure 13", "HIGH_RISK",
    "1. VERY WIDE (3651px / aspect 2.35:1) - 3 heatmaps horizontal.\n"
    "   Each heatmap has 15 columns (turns) with cell text at fontsize=8.\n"
    "   In each heatmap's width, 15 cells must fit - cells may be too narrow,\n"
    "   causing NUMBER OVERLAP (e.g., '85' and '90' in adjacent cells merging).\n"
    "   Y-axis labels have '\\n' replaced with space (e.g., 'N=INF (no heartbeat)'),\n"
    "   making them very long strings that crowd the heatmap area."))

# ── Figure 14: Latency ───────────────────────────────────────────────
print("\n[Figure 14] fig_heartbeat_latency.png (make_heartbeat_analysis.py)")
print("  Code: figsize=(8, 5), line plot 3 models with fill_between IQR band")
print("  Title: 2 lines")
print("  Status: OK - same layout as Figure 10, no crowding")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"SUMMARY: {len(issues)} figures with potential text overlap / rendering issues")
print("=" * 80)

severity_order = {"CRITICAL": 0, "HIGH_RISK": 1, "MODERATE": 2}
for fig, sev, desc in sorted(issues, key=lambda x: severity_order.get(x[1], 99)):
    print(f"\n  [{'*' * (4 - severity_order.get(sev, 99))}] {fig} [{sev}]")
    for line in desc.split("\n"):
        print(f"     {line}")

print(f"\nTotal problematic figures: {len(issues)} / 14")
