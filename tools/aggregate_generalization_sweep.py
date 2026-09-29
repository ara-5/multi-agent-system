"""Aggregate tools/run_generalization_sweep.py's per-(condition, seed) spatial-
generalization JSONs into bootstrap-CI'd curves of relative_improvement_pct
(trained vs. random-policy reward, normalized against the random baseline's
magnitude so it's comparable across position scales -- see that script's
docstring for why raw reward isn't) vs. position_scale, plus a pairwise
mi-vs-entropy comparison at each scale.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LABEL_RE = re.compile(r"^(?P<policy>bottleneck)_(?P<objective>entropy|mi)_seed(?P<seed>\d+)$")


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
    parser.add_argument("--results-dir", default="results/generalization_sweep")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_condition: dict[tuple[str, str], dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(results_dir.glob("*_generalization.json")):
        label = path.stem.replace("_generalization", "")
        match = LABEL_RE.match(label)
        if not match:
            continue
        with open(path) as f:
            per_scale = json.load(f)
        for record in per_scale:
            # Recomputed rather than trusting a stored relative_improvement_pct: an earlier version of
            # run_generalization_sweep.py had this sign inverted ((random - trained) instead of
            # (trained - random)) -- recomputing here from the raw reward_mean/random_reward_mean fields
            # is correct regardless of which formula produced the file on disk.
            record = dict(record)
            record["relative_improvement_pct"] = (
                (record["reward_mean"] - record["random_reward_mean"]) / abs(record["random_reward_mean"]) * 100
            )
            by_condition[(match["policy"], match["objective"])][record["scale"]].append(record)

    if not by_condition:
        raise SystemExit(f"No *_generalization.json files found under {results_dir}")

    print("=" * 100)
    print("Spatial generalization: relative improvement over random, by position scale (bootstrap 95% CI)")
    print("=" * 100)
    for (policy, objective), by_scale in sorted(by_condition.items()):
        n_seeds = len(next(iter(by_scale.values())))
        print(f"\n{policy} / {objective}  (n={n_seeds} seeds; scale=1.0 is the training distribution)")
        for scale in sorted(by_scale):
            records = by_scale[scale]
            rel = np.array([r["relative_improvement_pct"] for r in records])
            reward = np.array([r["reward_mean"] for r in records])
            rel_mean, rel_lo, rel_hi = bootstrap_ci(rel)
            print(f"  scale={scale:.1f}  reward={reward.mean():8.2f}  "
                  f"relative_improvement={rel_mean:6.1f}%  95% CI=[{rel_lo:6.1f}, {rel_hi:6.1f}]")

    print("\n" + "=" * 100)
    print("Pairwise: bottleneck/mi vs. bottleneck/entropy relative improvement, at each scale")
    print("=" * 100)
    mi_key, ent_key = ("bottleneck", "mi"), ("bottleneck", "entropy")
    if mi_key in by_condition and ent_key in by_condition:
        for scale in sorted(by_condition[mi_key]):
            if scale not in by_condition[ent_key]:
                continue
            a = np.array([r["relative_improvement_pct"] for r in by_condition[mi_key][scale]])
            b = np.array([r["relative_improvement_pct"] for r in by_condition[ent_key][scale]])
            diff, lo, hi, p = bootstrap_diff_ci(a, b)
            sig = "*" if p < 0.05 else " "
            print(f"\nscale={scale:.1f}  mi (mean={a.mean():.1f}%) vs. entropy (mean={b.mean():.1f}%)")
            print(f"  diff={diff:7.2f}pp  95% CI=[{lo:7.2f}, {hi:7.2f}]  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
