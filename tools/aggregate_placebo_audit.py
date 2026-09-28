"""Aggregate tools/placebo_audit_sweep.py's per-model *_causal_placebo.json
files into per-condition placebo-corrected statistics: the "message-specific
signal" (message-slice argmax-flip-rate minus velocity-slice placebo
flip-rate) that comm/causal_intervention.py's docstring says is the
trustworthy quantity, not the raw message flip-rate alone.

Why this is a separate script from tools/aggregate_seed_sweep.py: that
script's per-seed result.json files were written before
placebo_intervention_metrics existed (see placebo_audit_sweep.py's docstring
for why), so they don't have the placebo field. This reads the *_causal_
placebo.json files placebo_audit_sweep.py wrote directly instead, and joins
them back to (policy, objective, seed) via the filename convention
run_seed_sweep.py established (`{policy}_{objective}_seed{N}`)."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

LABEL_RE = re.compile(r"^(?P<policy>bottleneck|shared_trunk)_(?P<objective>none|entropy|mi|eccles)_seed(?P<seed>\d+)$")

COMPARISONS = [
    (("bottleneck", "none"), ("shared_trunk", "mi")),
    (("bottleneck", "mi"), ("shared_trunk", "mi")),
    (("bottleneck", "eccles"), ("shared_trunk", "eccles")),
    (("bottleneck", "entropy"), ("bottleneck", "mi")),
    (("bottleneck", "mi"), ("bottleneck", "eccles")),
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
    parser.add_argument("--results-dir", default="results/seed_sweep")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_condition: dict[tuple[str, str], list[dict]] = {}

    for path in sorted(results_dir.glob("*_causal_placebo.json")):
        label = path.stem.replace("_causal_placebo", "")
        match = LABEL_RE.match(label)
        if not match:
            continue
        with open(path) as f:
            data = json.load(f)
        if "placebo_argmax_flip_rate" not in data:
            continue
        message_flip = data["argmax_flip_rate"] * 100
        placebo_flip = data["placebo_argmax_flip_rate"] * 100
        record = {
            "seed": int(match["seed"]),
            "message_flip_pct": message_flip,
            "placebo_flip_pct": placebo_flip,
            "signal_pct": message_flip - placebo_flip,
        }
        by_condition.setdefault((match["policy"], match["objective"]), []).append(record)

    if not by_condition:
        raise SystemExit(f"No *_causal_placebo.json files with placebo data found under {results_dir}")

    print("=" * 100)
    print("Placebo-corrected causal signal per condition (bootstrap 95% CI, 10,000 resamples)")
    print("=" * 100)
    for (policy, objective), runs in sorted(by_condition.items()):
        n = len(runs)
        message = np.array([r["message_flip_pct"] for r in runs])
        placebo = np.array([r["placebo_flip_pct"] for r in runs])
        signal = np.array([r["signal_pct"] for r in runs])
        print(f"\n{policy} / {objective}  (n={n} seeds)")
        print(f"  message flip-rate %  mean={message.mean():6.2f}  {bootstrap_ci(message)[1:]}")
        print(f"  placebo flip-rate %  mean={placebo.mean():6.2f}  {bootstrap_ci(placebo)[1:]}")
        m, lo, hi = bootstrap_ci(signal)
        sig_note = "message flip-rate ABOVE placebo (real signal)" if lo > 0 else (
            "message flip-rate BELOW placebo (suspicious)" if hi < 0 else
            "CI includes 0 -- not distinguishable from placebo noise"
        )
        print(f"  message-specific signal (message - placebo), pp: mean={m:6.2f}  95% CI=[{lo:6.2f}, {hi:6.2f}]  -- {sig_note}")

    print("\n" + "=" * 100)
    print("Pairwise comparisons on message-specific signal")
    print("=" * 100)
    for cond_a, cond_b in COMPARISONS:
        if cond_a not in by_condition or cond_b not in by_condition:
            continue
        a = np.array([r["signal_pct"] for r in by_condition[cond_a]])
        b = np.array([r["signal_pct"] for r in by_condition[cond_b]])
        diff, lo, hi, p = bootstrap_diff_ci(a, b)
        sig = "*" if p < 0.05 else " "
        print(f"\n{cond_a[0]}/{cond_a[1]} (mean={a.mean():.2f}) vs. {cond_b[0]}/{cond_b[1]} (mean={b.mean():.2f})")
        print(f"  diff={diff:7.2f}  95% CI=[{lo:7.2f}, {hi:7.2f}]  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
