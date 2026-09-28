"""Run every (policy, objective) condition in comm/train_ablation.py across
many seeds, evaluating each trained model with comm/evaluate.py,
comm/analyze_communication.py, and comm/causal_intervention.py, and collecting
every run's metrics into one JSON file for tools/aggregate_seed_sweep.py.

Why this exists: the README's existing multi-seed results (3 seeds) are
already enough to show entropy-bonus's variance is real, but 3 seeds isn't
enough to put a trustworthy confidence interval on anything, or to say with
any rigor whether the MI-loss/Eccles-loss conditions are *reliably* different
from each other, not just different in this one 3-seed sample -- Colas et al.
2018 (arXiv:1806.08295, "How Many Random Seeds?") is directly about this
exact problem and recommends more seeds plus explicit power analysis rather
than reporting mean +/- std over a handful of runs as if it were a settled
result. This script doesn't do the full statistical-power-analysis workflow
that paper describes (that requires an assumed effect size up front, which we
don't have yet) -- it runs a fixed, larger seed budget and leaves the actual
significance testing to aggregate_seed_sweep.py's bootstrap CIs, which make
no distributional assumptions about the (possibly bimodal, per the
entropy-bonus collapse case) per-seed outcomes.

Conditions run (see comm/train_ablation.py's docstring for why the
shared_trunk rows matter most):
  bottleneck / {none, entropy, mi, eccles}
  shared_trunk / {mi, eccles}
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

CONDITIONS = [
    ("bottleneck", "none"),
    ("bottleneck", "entropy"),
    ("bottleneck", "mi"),
    ("bottleneck", "eccles"),
    ("shared_trunk", "mi"),
    ("shared_trunk", "eccles"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8, help="Seeds 0..N-1 for every condition.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--analyze-episodes", type=int, default=300)
    parser.add_argument("--causal-episodes", type=int, default=200)
    parser.add_argument("--out-dir", default="results/seed_sweep")
    parser.add_argument("--models-dir", default="models/seed_sweep")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip (condition, seed) pairs whose result JSON already exists -- for resuming.")
    parser.add_argument("--conditions", default=None,
                         help="Comma-separated policy:objective pairs to run instead of all of CONDITIONS -- "
                              "e.g. 'bottleneck:mi,shared_trunk:mi'. Lets multiple invocations of this script "
                              "run different conditions concurrently (one process per condition) instead of "
                              "one process working through all conditions serially.")
    parser.add_argument("--num-landmarks", type=int, default=None,
                         help="Passed through to train_ablation.py/evaluate.py/analyze_communication.py/ "
                              "causal_intervention.py -- for the vocabulary-scaling experiment. Unset (default) "
                              "uses the stock 3-landmark env.")
    args = parser.parse_args()

    conditions = CONDITIONS
    if args.conditions:
        conditions = [tuple(pair.split(":")) for pair in args.conditions.split(",")]

    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for policy, objective in conditions:
        for seed in range(args.seeds):
            label = f"{policy}_{objective}_seed{seed}"
            result_path = out_dir / f"{label}.json"
            if args.skip_existing and result_path.exists():
                print(f"Skipping {label} (already exists)")
                continue

            landmark_args = [] if args.num_landmarks is None else ["--num-landmarks", str(args.num_landmarks)]

            model_path = models_dir / label
            diagnostic_csv = out_dir / f"{label}_mi_diagnostic.csv"
            run([
                sys.executable, "comm/train_ablation.py",
                "--policy", policy, "--objective", objective,
                "--timesteps", str(args.timesteps), "--seed", str(seed),
                "--out", str(model_path), "--mi-diagnostic-log", str(diagnostic_csv),
                *landmark_args,
            ])
            with open(diagnostic_csv) as f:
                final_passive_mi_pct = float(list(csv.DictReader(f))[-1]["passive_mi_estimate_pct_of_max"])

            eval_out = subprocess.run(
                [sys.executable, "comm/evaluate.py", "--model", str(model_path),
                 "--episodes", str(args.eval_episodes), "--base-seed", "0", *landmark_args],
                check=True, capture_output=True, text=True,
            ).stdout
            reward_line = next(l for l in eval_out.splitlines() if l.startswith("Mean reward"))
            # "Mean reward over 20 episodes: -18.73 +/- 4.16"
            reward_mean, reward_std = reward_line.split(":")[1].strip().split(" +/- ")

            analyze_json = out_dir / f"{label}_analyze.json"
            run([
                sys.executable, "comm/analyze_communication.py", "--model", str(model_path),
                "--episodes", str(args.analyze_episodes), "--base-seed", "0",
                "--out", str(out_dir / f"{label}_confusion.png"), "--json-out", str(analyze_json),
                *landmark_args,
            ])
            with open(analyze_json) as f:
                analyze = json.load(f)

            causal_json = out_dir / f"{label}_causal.json"
            run([
                sys.executable, "comm/causal_intervention.py", "--model", str(model_path),
                "--episodes", str(args.causal_episodes), "--base-seed", "10000",
                "--json-out", str(causal_json), *landmark_args,
            ])
            with open(causal_json) as f:
                causal = json.load(f)

            result = {
                "policy": policy,
                "objective": objective,
                "num_landmarks": args.num_landmarks or 3,
                "seed": seed,
                "reward_mean": float(reward_mean),
                "reward_std": float(reward_std),
                "mutual_info_pct_of_max": analyze["mutual_info_bits"] / analyze["max_mutual_info_bits"] * 100,
                "message_entropy_pct_of_max": (
                    analyze["message_entropy_bits"] / analyze["max_message_entropy_bits"] * 100
                ),
                "argmax_flip_rate_pct": causal["argmax_flip_rate"] * 100,
                "mean_l1_vs_zero": causal["mean_l1_vs_zero"],
                "mean_kl_vs_zero": causal["mean_kl_vs_zero"],
                "final_passive_mi_diagnostic_pct": final_passive_mi_pct,
            }
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[{label}] MI={result['mutual_info_pct_of_max']:.1f}% "
                  f"flip_rate={result['argmax_flip_rate_pct']:.1f}% reward={result['reward_mean']:.2f}")


if __name__ == "__main__":
    main()
