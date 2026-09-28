"""Aggregate tools/run_seed_sweep.py's per-(condition, seed) JSON results into
per-condition summary statistics with bootstrap confidence intervals, and a
pairwise bootstrap test between conditions of interest.

Why bootstrap rather than mean +/- std or a t-test: this repo's own README
already found a bimodal per-seed outcome (the entropy-bonus condition:
99.7%, 0.0%, 55.3%) -- a distribution a t-test's normality assumption doesn't
fit, and where a std-dev alone is a misleading summary (it doesn't distinguish
"consistently mediocre" from "usually great, occasionally catastrophic").
A percentile bootstrap on the mean, and on the difference of two conditions'
means, makes no distributional assumption and is the approach Colas et al.
2018 (arXiv:1806.08295) recommend for exactly this kind of RL seed-variance
comparison, over reporting a bare mean +/- std from a handful of seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    "mutual_info_pct_of_max", "message_entropy_pct_of_max",
    "argmax_flip_rate_pct", "reward_mean",
]

COMPARISONS = [
    (("bottleneck", "entropy"), ("bottleneck", "mi")),
    (("bottleneck", "entropy"), ("bottleneck", "eccles")),
    (("bottleneck", "mi"), ("bottleneck", "eccles")),
    (("bottleneck", "mi"), ("shared_trunk", "mi")),
    (("bottleneck", "eccles"), ("shared_trunk", "eccles")),
]


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, rng=None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, rng=None) -> tuple[float, float, float, float]:
    """Returns (mean_diff, ci_low, ci_high, p_two_sided) for mean(a) - mean(b),
    via a percentile bootstrap on the difference and a permutation-style
    two-sided p-value (fraction of bootstrap resamples where the sign of the
    difference flips relative to the observed one)."""
    rng = rng or np.random.default_rng(0)
    diffs = np.array([
        rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n_resamples)
    ])
    observed = a.mean() - b.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # min(P(diff>=0), P(diff<=0)) double-counts any point mass exactly at 0 (common
    # with very small n, where bootstrap resamples collide), so the raw product can
    # exceed 1 -- clip to keep this a valid two-sided p-value.
    p_value = min(1.0, 2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
    return float(observed), float(ci_low), float(ci_high), float(p_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/seed_sweep")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_condition: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(results_dir.glob("*_seed*.json")):
        if "_analyze" in path.stem or "_causal" in path.stem:
            continue
        with open(path) as f:
            r = json.load(f)
        by_condition.setdefault((r["policy"], r["objective"]), []).append(r)

    if not by_condition:
        raise SystemExit(f"No per-seed result JSONs found under {results_dir} -- run tools/run_seed_sweep.py first.")

    print("=" * 100)
    print("Per-condition summary (bootstrap 95% CI on the mean, 10,000 resamples)")
    print("=" * 100)
    for (policy, objective), runs in by_condition.items():
        n = len(runs)
        print(f"\n{policy} / {objective}  (n={n} seeds)")
        per_seed_mi = sorted(round(r["mutual_info_pct_of_max"], 1) for r in runs)
        print(f"  per-seed MI% (sorted, for spotting bimodality a mean can hide): {per_seed_mi}")
        for metric in METRICS:
            values = np.array([r[metric] for r in runs])
            mean, lo, hi = bootstrap_ci(values)
            print(f"  {metric:32s} mean={mean:8.2f}  95% CI=[{lo:7.2f}, {hi:7.2f}]  std={values.std():.2f}")

    print("\n" + "=" * 100)
    print("Pairwise comparisons (bootstrap 95% CI + two-sided bootstrap p-value on the difference of means)")
    print("=" * 100)
    for cond_a, cond_b in COMPARISONS:
        if cond_a not in by_condition or cond_b not in by_condition:
            continue
        runs_a, runs_b = by_condition[cond_a], by_condition[cond_b]
        print(f"\n{cond_a[0]}/{cond_a[1]}  vs.  {cond_b[0]}/{cond_b[1]}  (n={len(runs_a)} vs n={len(runs_b)})")
        for metric in ["mutual_info_pct_of_max", "argmax_flip_rate_pct"]:
            a = np.array([r[metric] for r in runs_a])
            b = np.array([r[metric] for r in runs_b])
            diff, lo, hi, p = bootstrap_diff_ci(a, b)
            sig = "*" if p < 0.05 else " "
            print(f"  {metric:28s} diff={diff:8.2f}  95% CI=[{lo:7.2f}, {hi:7.2f}]  p={p:.4f} {sig}")

    print("\n" + "=" * 100)
    print("Passive in-training MI diagnostic (mi_diagnostic_callback.py) vs. post-hoc ground truth")
    print("=" * 100)
    all_runs = [r for runs in by_condition.values() for r in runs]
    passive = np.array([r["final_passive_mi_diagnostic_pct"] for r in all_runs])
    ground_truth_mi = np.array([r["mutual_info_pct_of_max"] for r in all_runs])
    flip_rate = np.array([r["argmax_flip_rate_pct"] for r in all_runs])
    corr_vs_mi = float(np.corrcoef(passive, ground_truth_mi)[0, 1])
    corr_vs_causal = float(np.corrcoef(passive, flip_rate)[0, 1])
    print(f"n={len(all_runs)} runs across all conditions")
    print(f"  Pearson r(passive diagnostic, post-hoc confusion-matrix MI):        {corr_vs_mi:.3f}")
    print(f"  Pearson r(passive diagnostic, causal argmax-flip rate):             {corr_vs_causal:.3f}")
    print("  A high first correlation and a much lower second one would show the free training-time estimate "
          "is a good proxy for the correlational metric but NOT the causal one -- i.e. it can tell you the "
          "message looks informative without telling you the listener is actually using it (see "
          "shared_trunk_mi_policy.py's and causal_intervention.py's docstrings for why that gap can exist).")

    print("\nNote: with n<20 seeds per condition, treat these CIs/p-values as suggestive, not conclusive -- "
          "see this script's docstring and Colas et al. 2018 (arXiv:1806.08295) on seed counts needed for "
          "adequately powered RL comparisons. Widen --seeds in run_seed_sweep.py before citing these numbers "
          "in anything you intend to publish.")


if __name__ == "__main__":
    main()
