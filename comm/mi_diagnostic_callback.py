"""A passive, real-time "communication health monitor" -- not a loss, just a
logger. Every SB3 PPO run already collects a rollout buffer of (observation,
action) pairs before each policy update; this callback reads the message
channel out of that buffer at zero extra environment cost (no held-out
episodes, no extra rollouts) and logs mi_ppo.py's RIM/IMSAT batch-level
mutual-information estimate to a CSV, timestep by timestep, for *any* run --
including train_baseline.py's/`--objective none`/`--objective entropy`/
`--objective eccles` runs that never use this quantity as a loss at all.

Why this might be worth having as a feature in its own right, not just an
ablation tool: comm/analyze_communication.py's confusion-matrix MI is the
trustworthy ground-truth measurement in this repo (see its docstring), but
it's expensive relative to training -- it needs a separate deterministic
rollout of >=200 held-out episodes, run *after* training finishes. If this
callback's free, in-training running estimate reliably tracks (or, better,
predicts) where analyze_communication.py's post-hoc number will land, that's
a cheap early-warning signal: you could tell a communication protocol is
converging to a degenerate one (like train_baseline.py's plateau, or the
entropy-bonus condition's occasional collapse) from the training log alone,
without waiting for training to finish and running a separate analysis pass.
If it *doesn't* track the ground truth reliably (e.g. under
shared_trunk_mi_policy.py's ablation, where the message can satisfy this
batch estimator while the listener still doesn't causally depend on it --
see causal_intervention.py), that's an equally important finding: it would
mean this cheap proxy is only trustworthy under an architecture that already
enforces a real bottleneck, and cannot substitute for the causal check.
tools/aggregate_seed_sweep.py's runs are what actually test which of these is
true -- this file only collects the data needed to check.
"""
import csv
import os

import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from common.comm_audit import message_mutual_info_estimate


class MIDiagnosticCallback(BaseCallback):
    def __init__(self, csv_path: str, max_mutual_info_bits: float, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.max_mutual_info_nats = max_mutual_info_bits * 0.6931471805599453  # ln(2), bits -> nats

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "passive_mi_estimate_pct_of_max"])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        with th.no_grad():
            observations = self.model.rollout_buffer.observations
            obs_tensor = th.as_tensor(observations.reshape(-1, observations.shape[-1]), device=self.model.device)
            features = self.model.policy.extract_features(obs_tensor)
            message_logits = self.model.policy.message_logits(features)
            estimate = message_mutual_info_estimate(message_logits).item()
        pct_of_max = estimate / self.max_mutual_info_nats * 100
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, pct_of_max])
        self.logger.record("train/passive_mi_diagnostic_pct_of_max", pct_of_max)
