"""Check all paper figures for potential text overlap / rendering issues."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "data", "figures")

# All figures referenced in the paper
PAPER_FIGURES = {
    "Figure 1": "fig1_passrate_heatmap.png",
    "Figure 2": "fig2_spearman_bar.png",
    "Figure 3": "fig3_token_variance.png",
    "Figure 4": "fig4_frontend_ranking.png",
    "Figure 5": "fig5_case_level_grid.png",
    "Figure 6": "tradeoff_curve.png",
    "Figure 7": "ablation_overall.png",
    "Figure 8": "ablation_by_type.png",
    "Figure 9": "entropy_curve.png",
    "Figure 10": "fig_heartbeat_passrate_by_context.png",
    "Figure 11": "fig_heartbeat_error_propagation.png",
    "Figure 12": "fig_heartbeat_n_overall.png",
    "Figure 13": "fig_heartbeat_turn_heatmap.png",
    "Figure 14": "fig_heartbeat_latency.png",
}

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[warn] PIL not installed, checking file existence only", file=sys.stderr)

print("=" * 70)
print("Paper Figure Audit Report")
print("=" * 70)

issues = []

for fig_name, filename in sorted(PAPER_FIGURES.items()):
    path = os.path.join(FIGURES_DIR, filename)
    print(f"\n{fig_name}: {filename}", end="")

    if not os.path.exists(path):
        print(f"  *** MISSING ***")
        issues.append((fig_name, "FILE_MISSING", f"{filename} not found"))
        continue

    if HAS_PIL:
        img = Image.open(path)
        w, h = img.size
        aspect = w / h if h > 0 else 0
        print(f"  [{w}x{h}, aspect={aspect:.2f}]")

        # Flag potential issues based on dimensions and code analysis
        # Figure 5: 5 subplots side by side - very wide, risk of label overlap
        if fig_name == "Figure 5":
            if w > 2000:
                print(f"  ! WARNING: Very wide figure ({w}px). Subplot x-labels may overlap.")
                issues.append((fig_name, "POTENTIAL_OVERLAP", f"Very wide ({w}px), 5 subplots may have x-label overlap"))
            if h < 400:
                print(f"  ! WARNING: Short height ({h}px). Y-labels (27 cases) may be cramped.")
                issues.append((fig_name, "POTENTIAL_CRAMPED", f"Short height ({h}px) for 27 case labels"))

        # Figure 8: ablation by type - 9 categories with grouped bars
        if fig_name == "Figure 8":
            if w < 900:
                print(f"  ! WARNING: Narrow width ({w}px). 9 category labels may overlap.")
                issues.append((fig_name, "POTENTIAL_OVERLAP", f"Narrow width ({w}px) for 9 rotated category labels"))

        # Figure 13: heatmap - 6 N-values x 15 turns, 3 models side by side
        if fig_name == "Figure 13":
            if w < 1800:
                print(f"  ! WARNING: 3 heatmaps side-by-side in {w}px. Cell text may overlap.")
                issues.append((fig_name, "POTENTIAL_OVERLAP", f"3 heatmaps in {w}px width, cell annotations may crowd"))

        # Figure 2: spearman bar - 10 pairs with long y-labels
        if fig_name == "Figure 2":
            print(f"  * Note: Y-axis uses multi-line labels (model_a\\nvs\\nmodel_b)")
            if h < 500:
                issues.append((fig_name, "POTENTIAL_CRAMPED", f"Height {h}px may be tight for 10 multi-line y-labels"))

        # General: flag unusually small figures
        if w < 600 or h < 400:
            print(f"  ! WARNING: Small figure - text may be hard to read when embedded.")
            issues.append((fig_name, "SMALL_SIZE", f"Small dimension {w}x{h}"))

    else:
        print("  [exists]")

print("\n" + "=" * 70)
print(f"SUMMARY: {len(issues)} potential issue(s) found")
print("=" * 70)
if issues:
    for fig, severity, desc in issues:
        print(f"  [{severity}] {fig}: {desc}")
