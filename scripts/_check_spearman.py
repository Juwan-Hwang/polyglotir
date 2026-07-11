#!/usr/bin/env python
"""Verify Spearman significance claims."""
import json, math

data = json.load(open("data/processed/phase2/phase2_spearman.json", encoding="utf-8"))

n = data['n_data_points']
df = n - 2

print("n_data_points per pair: {}".format(n))
print("valid: {}".format(data['valid']))
print()
print("n={}, df={}".format(n, df))
print("t-critical (alpha=0.05, two-tailed, df={}): ~2.017".format(df))
print("t-critical (alpha=0.01, two-tailed, df={}): ~2.696".format(df))
print()

print("{:40s} {:>7s} {:>8s} {:>8s} {:>8s}".format("pair", "rho", "t-stat", "p<0.05", "p<0.01"))
print("-" * 75)

for p in data['pairs']:
    rho = p['spearman_rho']
    if abs(rho) < 1:
        t_stat = rho * math.sqrt(df / (1 - rho**2))
    else:
        t_stat = float('inf')
    sig05 = "YES" if abs(t_stat) > 2.017 else "no"
    sig01 = "YES" if abs(t_stat) > 2.696 else "no"
    pair_name = "{} vs {}".format(p['model_a'], p['model_b'])
    print("{:40s} {:7.4f} {:8.3f} {:>8s} {:>8s}".format(pair_name, rho, t_stat, sig05, sig01))

print()
print("=== Original roadmap threshold ===")
print("SILP doc says: '150+ points for Spearman'")
print("Current data points: {}".format(n))
print("Current code threshold: 30 (lowered from 150 in prior conversation)")
print()

print("=== Spearman critical values (alpha=0.05, two-tailed) ===")
for nn in [20, 25, 30, 35, 40, 45, 50, 100, 150]:
    if nn <= 30:
        cv = {20: 0.450, 25: 0.400, 30: 0.364}.get(nn, 0.364)
    else:
        cv = 1.96 / math.sqrt(nn - 1)
    print("  n={:3d}: critical rho ~= {:.4f}".format(nn, cv))
