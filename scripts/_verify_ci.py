#!/usr/bin/env python
"""Verify CI and Spearman critical value calculations."""
import math

print("=== CI half-width for p=0.80, 95% confidence ===")
for n_cell in [9, 15, 20, 25, 30, 45]:
    se = math.sqrt(0.8 * 0.2 / n_cell)
    hw = 1.96 * se * 100  # in percentage points
    print("  n={:2d}/cell: +-{:5.1f}pp".format(n_cell, hw))

print()
print("=== Marginal improvement ===")
prev = None
for n_cell in [9, 15, 20, 25, 30, 45]:
    se = math.sqrt(0.8 * 0.2 / n_cell)
    hw = 1.96 * se * 100
    if prev:
        print("  {}->{}: CI narrows by {:5.1f}pp ({:.0f}% improvement)".format(
            prev, n_cell, prev - hw, (prev - hw) / prev * 100))
    prev = hw

print()
print("=== Spearman critical values (alpha=0.05, two-tailed) ===")
for n_total in [45, 75, 100, 150, 225]:
    cv = 1.96 / math.sqrt(n_total - 1)
    print("  n={:3d} total points: critical rho ~= {:.4f}".format(n_total, cv))

print()
print("=== Current weakest pair: rho=0.367 ===")
for n_total in [45, 75, 150, 225]:
    cv = 1.96 / math.sqrt(n_total - 1)
    print("  n={:3d}: 0.367 vs {:.4f} -> {} margin".format(
        n_total, cv, "+{:.3f}".format(0.367 - cv)))
