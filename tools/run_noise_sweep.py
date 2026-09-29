"""Run comm/noise_robustness.py's message-corruption sweep against every
already-trained model from tools/run_seed_sweep.py, without retraining
anything -- this reuses the exact same 6-seeds-per-condition models the
paper's MI and causal-audit numbers come from, so the noise-robustness result
is directly comparable to them rather than a fresh, differently-seeded run.

Writes one JSON per (condition, seed) to --out-dir, each a list of
per-noise-level dicts (see comm/noise_robustness.py's docstring for what's
measured at each level). tools/aggregate_noise_sweep.py turns these into
bootstrap-CI'd per-condition curves.
"""
import argparse
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
    parser.add_argument("--seeds", type=int, default=6, help="Seeds 0..N-1 for every condition.")
    parser.add_argument("--noise-levels", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--models-dir", default="models/seed_sweep",
                         help="Where tools/run_seed_sweep.py already saved trained models.")
    parser.add_argument("--out-dir", default="results/noise_sweep")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--conditions", default=None,
                         help="Comma-separated policy:objective pairs, e.g. 'bottleneck:mi,shared_trunk:mi', "
                              "for running conditions concurrently across multiple invocations.")
    args = parser.parse_args()

    conditions = CONDITIONS
    if args.conditions:
        conditions = [tuple(pair.split(":")) for pair in args.conditions.split(",")]

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for policy, objective in conditions:
        for seed in range(args.seeds):
            label = f"{policy}_{objective}_seed{seed}"
            model_path = models_dir / label
            if not model_path.with_suffix(".zip").exists():
                print(f"Skipping {label}: no trained model at {model_path}.zip")
                continue

            result_path = out_dir / f"{label}_noise.json"
            if args.skip_existing and result_path.exists():
                print(f"Skipping {label} (already exists)")
                continue

            run([
                sys.executable, "comm/noise_robustness.py",
                "--model", str(model_path),
                "--noise-levels", args.noise_levels,
                "--episodes", str(args.episodes),
                "--json-out", str(result_path),
            ])

    print(f"\nDone. Results in {out_dir}/*_noise.json")


if __name__ == "__main__":
    main()
