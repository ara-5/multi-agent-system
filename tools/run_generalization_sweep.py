"""Spatial-generalization sweep: evaluate already-trained models at reset
position ranges wider than the uniform(-1,+1) box they were trained under
(--position-scale, see common/scaled_speaker_listener.py), to test whether
the *movement* policy -- not the message protocol, which is scale-invariant
by construction here, see that module's docstring -- generalizes to
landmark/agent geometries it never saw in training.

Reuses the models/vocab_sweep_n3/ models (bottleneck/{entropy,mi}, 4 seeds
each) rather than the main stock-env seed_sweep models, because
--position-scale only works through ScaledScenario, and mixing a
ScaledScenario-trained model's baseline (scale=1.0) numbers with the stock
env's would compare across two different speaker-observation encodings (see
common/scaled_speaker_listener.py's docstring) -- not apples to apples.

Raw reward isn't directly comparable across scales: distance-based reward
mechanically gets larger in magnitude as the box widens, regardless of policy
quality (a random policy's reward gets much worse too). This script also runs
a random-policy baseline at each scale and reports relative_improvement_pct =
(trained_reward - random_reward) / abs(random_reward) * 100, which is scale-
normalized: 0% means no better than random, positive means better than random
(reward is less negative than random's), 100% would mean zero cost (reward=0)
at that scale.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CONDITIONS = [
    ("bottleneck", "entropy"),
    ("bottleneck", "mi"),
]


def mean_reward(cmd: list[str]) -> float:
    print("+", " ".join(cmd))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("Mean reward"))
    return float(line.split(":")[1].strip().split(" +/- ")[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4, help="Seeds 0..N-1, matching models/vocab_sweep_n3/.")
    parser.add_argument("--scales", default="1.0,1.5,2.0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=30_000,
                         help="Disjoint from every other script's eval seeds in this repo.")
    parser.add_argument("--models-dir", default="models/vocab_sweep_n3")
    parser.add_argument("--out-dir", default="results/generalization_sweep")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_reward_by_scale = {}
    for scale in scales:
        random_reward_by_scale[scale] = mean_reward([
            sys.executable, "comm/evaluate.py", "--model", "", "--num-landmarks", "3",
            "--position-scale", str(scale), "--episodes", str(args.random_episodes),
            "--base-seed", str(args.base_seed),
        ])
        print(f"[random baseline] scale={scale}  reward={random_reward_by_scale[scale]:.2f}")

    for policy, objective in CONDITIONS:
        for seed in range(args.seeds):
            label = f"{policy}_{objective}_seed{seed}"
            model_path = models_dir / label
            if not model_path.with_suffix(".zip").exists():
                print(f"Skipping {label}: no trained model at {model_path}.zip")
                continue

            records = []
            for scale in scales:
                trained_reward = mean_reward([
                    sys.executable, "comm/evaluate.py", "--model", str(model_path), "--num-landmarks", "3",
                    "--position-scale", str(scale), "--episodes", str(args.episodes),
                    "--base-seed", str(args.base_seed),
                ])
                random_reward = random_reward_by_scale[scale]
                relative_improvement_pct = (trained_reward - random_reward) / abs(random_reward) * 100
                records.append({
                    "scale": scale,
                    "reward_mean": trained_reward,
                    "random_reward_mean": random_reward,
                    "relative_improvement_pct": relative_improvement_pct,
                })
                print(f"[{label}] scale={scale}  reward={trained_reward:.2f}  "
                      f"relative_improvement={relative_improvement_pct:.1f}%")

            with open(out_dir / f"{label}_generalization.json", "w") as f:
                json.dump(records, f, indent=2)

    print(f"\nDone. Results in {out_dir}/*_generalization.json")


if __name__ == "__main__":
    main()
