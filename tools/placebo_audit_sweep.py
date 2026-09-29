"""Re-run comm/causal_intervention.py's placebo-controlled audit against every
model tools/run_seed_sweep.py already trained and saved, without retraining
anything.

Why this is a separate pass instead of being folded into run_seed_sweep.py:
common/comm_audit.py's placebo_intervention_metrics() and
comm/causal_intervention.py's --json-out placebo field were added *after*
most of the main sweep had already finished (or was mid-run) -- see this
repo's history for why (editing files a live background sweep imports is
risky; this script only reads already-saved .zip models, so it can't race
with anything). Re-running training would cost another ~30-60 minutes for no
reason: the model weights don't change, only the (cheap, inference-only)
audit does.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models/seed_sweep")
    parser.add_argument("--out-dir", default="results/seed_sweep")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip a model if its output JSON already has a 'mean_l1_vs_alternatives' field -- "
                              "for resuming a sweep in bounded batches instead of one long-running process.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)

    for model_path in sorted(models_dir.glob("*.zip")):
        label = model_path.stem
        out_json = out_dir / f"{label}_causal_placebo.json"
        if args.skip_existing and out_json.exists():
            with open(out_json) as f:
                existing = json.load(f)
            if "mean_l1_vs_alternatives" in existing and existing.get("episodes") == args.episodes:
                print(f"- {label} (already up to date, skipping)")
                continue
        print(f"+ {label}")
        subprocess.run([
            sys.executable, "comm/causal_intervention.py",
            "--model", str(model_path.with_suffix("")),
            "--episodes", str(args.episodes), "--base-seed", str(args.base_seed),
            "--json-out", str(out_json),
        ], check=True, capture_output=True, text=True)
        with open(out_json) as f:
            result = json.load(f)
        print(f"  flip_rate={result['argmax_flip_rate'] * 100:.1f}%  "
              f"placebo={result.get('placebo_argmax_flip_rate', float('nan')) * 100:.1f}%  "
              f"signal={(result['argmax_flip_rate'] - result.get('placebo_argmax_flip_rate', 0)) * 100:+.1f}pp")


if __name__ == "__main__":
    main()
