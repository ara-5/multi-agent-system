"""Aggregate tools/run_noise_sweep.py's per-(condition, seed) noise-sweep
JSONs into per-condition degradation curves (reward and decision-flip rate
vs. message-corruption probability) with bootstrap 95% CIs, plus pairwise
comparisons on decision_flip_rate at the highest noise level -- the
"how much does corruption at a given rate actually change the listener's
decision" comparison across conditions.

Two curves are reported per condition, and they answer different questions
(see comm/noise_robustness.py's docstring): reward_mean tends to be nearly
flat across noise levels for every condition in this environment, because
distance-based reward integrated over 25 steps averages out occasional
corrupted steps -- this is itself worth reporting plainly rather than
silently dropping the reward curve because it isn't the dramatic collapse a
naive prediction might expect. decision_flip_rate is the more sensitive
metric: it measures whether the corruption actually changed what the
listener would have done at that instant, independent of whether the
resulting trajectory recovers.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LABEL_RE = re.compile(r"^(?P<policy>bottleneck|shared_trunk)_(?P<objective>none|entropy|mi|eccles)_seed(?P<seed>\d+)$")

COMPARISONS = [
    (("bottleneck", "none"), ("bottleneck", "mi")),
    (("bottleneck", "none"), ("shared_trunk", "mi")),
    (("bottleneck", "mi"), ("shared_trunk", "mi")),
    (("bottleneck", "mi"), ("bottleneck", "eccles")),
    (("shared_trunk", "mi"), ("shared_trunk", "eccles")),
]


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, rng=None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, rng=None):
    rng = rng or np.random.default_rng(0)
    diffs = np.array([
        rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n_resamples)
    ])
    observed = a.mean() - b.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = min(1.0, 2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
    return float(observed), float(ci_low), float(ci_high), float(p_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/noise_sweep")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    # by_condition[(policy, objective)][noise_pct] = list of per-seed dicts
    by_condition: dict[tuple[str, str], dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(results_dir.glob("*_noise.json")):
        label = path.stem.replace("_noise", "")
        match = LABEL_RE.match(label)
        if not match:
            continue
        import json
        with open(path) as f:
            per_noise_level = json.load(f)
        for record in per_noise_level:
            by_condition[(match["policy"], match["objective"])][record["noise_pct"]].append(record)

    if not by_condition:
        raise SystemExit(f"No *_noise.json files found under {results_dir}")

    print("=" * 100)
    print("Noise-robustness degradation curves per condition (bootstrap 95% CI, 10,000 resamples)")
    print("=" * 100)
    for (policy, objective), by_noise in sorted(by_condition.items()):
        n_seeds = len(next(iter(by_noise.values())))
        print(f"\n{policy} / {objective}  (n={n_seeds} seeds)")
        print(f"  {'noise':>7}  {'reward mean':>18}  {'decision flip-rate %':>24}")
        for noise_pct in sorted(by_noise):
            records = by_noise[noise_pct]
            rewards = np.array([r["reward_mean"] for r in records])
            r_mean, r_lo, r_hi = bootstrap_ci(rewards)
            flips = [r["decision_flip_rate"] for r in records if r["decision_flip_rate"] is not None]
            if flips:
                flip_arr = np.array(flips) * 100
                f_mean, f_lo, f_hi = bootstrap_ci(flip_arr)
                flip_str = f"{f_mean:6.1f}  [{f_lo:5.1f}, {f_hi:5.1f}]"
            else:
                flip_str = "n/a (noise=0)"
            print(f"  {noise_pct:6.1f}%  {r_mean:7.2f} [{r_lo:7.2f},{r_hi:7.2f}]  {flip_str}")

    print("\n" + "=" * 100)
    print("Pairwise comparisons on decision_flip_rate at the highest noise level")
    print("=" * 100)
    max_noise = max(noise for by_noise in by_condition.values() for noise in by_noise)
    for cond_a, cond_b in COMPARISONS:
        if cond_a not in by_condition or cond_b not in by_condition:
            continue
        a = np.array([r["decision_flip_rate"] for r in by_condition[cond_a][max_noise]
                       if r["decision_flip_rate"] is not None]) * 100
        b = np.array([r["decision_flip_rate"] for r in by_condition[cond_b][max_noise]
                       if r["decision_flip_rate"] is not None]) * 100
        if len(a) == 0 or len(b) == 0:
            continue
        diff, lo, hi, p = bootstrap_diff_ci(a, b)
        sig = "*" if p < 0.05 else " "
        print(f"\n{cond_a[0]}/{cond_a[1]} (mean={a.mean():.1f}%) vs. {cond_b[0]}/{cond_b[1]} (mean={b.mean():.1f}%) "
              f"@ noise={max_noise:.0f}%")
        print(f"  diff={diff:7.2f}pp  95% CI=[{lo:7.2f}, {hi:7.2f}]  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
