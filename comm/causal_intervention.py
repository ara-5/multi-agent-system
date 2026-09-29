"""Quantify, across many held-out episodes, what web_demo/index.html's "force
the message" widget only shows one frame at a time: does the listener's
movement actually depend on the received message, or does it just happen to
correlate with the target the way analyze_communication.py's confusion-matrix
mutual information measures?

Why this is a different question from MI: analyze_communication.py checks
whether message and target co-occur in a consistent pattern -- a purely
observational, correlational test. Lowe et al. 2019 (arXiv:1903.05168, the
paper this whole repo's "reward-curve deception" story is an instance of)
specifically warn that such observational metrics can be misleading: nothing
about a target<->message correlation proves the *listener's action* is
causally driven by the message rather than by some other path that happens to
correlate with it too (in shared_trunk_mi_policy.py's ablation architecture,
that other path is literal -- the shared trunk has direct, unrestricted access
to the raw target). The listening-side counterpart, "causal influence of
communication" (CIC), comes from Jaques et al. 2019 (arXiv:1810.08647) and is
used explicitly as a positive-listening training loss (not just a metric, as
here) in Eccles et al. 2019 (arXiv:1912.05676, Sec 3.2, Eq. 4-8) -- see
eccles_ppo.py's docstring for the exact equations this script's estimator
follows (Eq. 5's CIC-as-KL-divergence and Eq. 8's L1 form), adapted from their
multi-timestep recurrent setting to this repo's single-step feedforward one.

For each of --episodes held-out rollouts, at each step this script takes the
policy's real message and movement action, then recomputes the movement
action for (a) the message zeroed out (Eccles' x'_t) and (b) every other
possible discrete message, holding the rest of the listener's observation
fixed -- exactly bottleneck_policy.py's / shared_trunk_mi_policy.py's
`movement_logits_with_message_override`, the same operation
web_demo/index.html's `listenerMovementForMessage` does in JS for one frame.
It reports:
  - argmax_flip_rate: fraction of steps where forcing a *different* message
    than the one actually sent changes the greedy movement action -- 0 means
    movement never depends on which message was sent, regardless of what MI
    says about the message.
  - mean_l1_vs_zero / mean_kl_vs_zero: the size of that dependence against the
    zero-message counterfactual specifically (Eccles' Eq. 8 / Eq. 5 estimators,
    reported as metrics here rather than losses).
A real, causally load-bearing protocol should score high on both this and
analyze_communication.py's MI; a shortcut (message correlates with target via
some other path, as in the shared-trunk-mi ablation) can score high on MI
while scoring near-zero here.

The actual metric computation lives in common/comm_audit.py's
causal_intervention_metrics(), which only needs a stream of (features,
real_message) pairs -- this file's job is just the simple_speaker_listener-
specific plumbing that produces that stream (running episodes, extracting
features, reading the real action) so the same metric function can be reused
against a different environment by writing a different stream, without
touching the metric logic itself.

This script also runs common/comm_audit.py's placebo_intervention_metrics()
as a control: the identical intervention procedure applied to the listener's
own velocity (the first 2 entries of its observation, per MPE's layout --
unrelated to the message) instead of the message slice. See that function's
docstring for why a message-slice flip rate alone isn't yet evidence of
message-specific causal reliance without this comparison -- a real protocol
should show a message-slice flip rate well above the placebo's; if the two
are close, the "real" result mostly reflects general policy brittleness to
any input perturbation, not the message specifically.
"""
import argparse
import json

import numpy as np
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from common.comm_audit import (
    causal_intervention_metrics,
    causal_intervention_metrics_by_target,
    placebo_intervention_metrics,
)
from common.joint_env import JointPolicyEnv
from common.scaled_speaker_listener import parallel_env as scaled_parallel_env

_VELOCITY_SLICE = (0, 2)  # self_vel is always the first 2 entries of the listener's own observation


def _feature_message_stream(env, model, episodes: int, base_seed: int):
    policy = model.policy
    for episode in range(episodes):
        obs, _info = env.reset(seed=base_seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_tensor = obs_as_tensor(np.expand_dims(obs, 0), policy.device)
            features = policy.extract_features(obs_tensor)
            action, _ = model.predict(obs, deterministic=True)
            yield features, int(action[0])
            obs, _reward, terminated, truncated, _info = env.step(action)


def _feature_message_target_stream(env, model, episodes: int, base_seed: int, speaker_dim: int):
    # The target is recoverable from the speaker's own observation slice without any extra
    # plumbing: SB3's default feature extractor for a Box obs space is FlattenExtractor (identity
    # flatten), so `features` here is just the raw observation -- the same
    # argmax(obs[:speaker_dim]) analyze_communication.py uses to read the true target off the
    # speaker's one-hot(-ish) observation.
    policy = model.policy
    for episode in range(episodes):
        obs, _info = env.reset(seed=base_seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_tensor = obs_as_tensor(np.expand_dims(obs, 0), policy.device)
            features = policy.extract_features(obs_tensor)
            target = int(features[0, :speaker_dim].argmax().item())
            action, _ = model.predict(obs, deterministic=True)
            yield features, int(action[0]), target
            obs, _reward, terminated, truncated, _info = env.step(action)


def _feature_slice_stream(env, model, episodes: int, base_seed: int, speaker_dim: int, start: int, end: int):
    policy = model.policy
    for episode in range(episodes):
        obs, _info = env.reset(seed=base_seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_tensor = obs_as_tensor(np.expand_dims(obs, 0), policy.device)
            features = policy.extract_features(obs_tensor)
            real_slice_value = features[:, speaker_dim + start : speaker_dim + end].clone()
            yield features, real_slice_value
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=10_000, help="Disjoint from analyze_communication.py's "
                         "default --base-seed 0, so this script's episodes are genuinely held-out from it.")
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-landmarks", type=int, default=None,
                         help="Must match what the model was trained with -- see train_ablation.py.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    model = PPO.load(args.model)
    policy = model.policy
    if not hasattr(policy, "movement_logits_with_message_override"):
        raise SystemExit(
            f"{type(policy).__name__} doesn't expose movement_logits_with_message_override -- "
            "this script only works with SpeakerListenerBottleneckPolicy or SharedTrunkAuxLossPolicy."
        )

    if args.num_landmarks is None:
        pz_env_fn = lambda: simple_speaker_listener_v4.parallel_env(
            max_cycles=args.max_cycles, continuous_actions=False,
        )
    else:
        pz_env_fn = lambda: scaled_parallel_env(
            num_landmarks=args.num_landmarks, max_cycles=args.max_cycles, continuous_actions=False,
        )
    env = JointPolicyEnv(pz_env_fn)
    num_messages = int(env.action_space.nvec[0])
    speaker_dim = env.obs_dims[0]

    stream = _feature_message_stream(env, model, args.episodes, args.base_seed)
    metrics = causal_intervention_metrics(policy, stream, num_messages)

    by_target_stream = _feature_message_target_stream(env, model, args.episodes, args.base_seed, speaker_dim)
    by_target = causal_intervention_metrics_by_target(policy, by_target_stream, num_messages)

    placebo_supported = hasattr(policy, "movement_logits_with_slice_override")
    if placebo_supported:
        slice_stream = _feature_slice_stream(
            env, model, args.episodes, args.base_seed, speaker_dim, *_VELOCITY_SLICE
        )
        placebo_metrics = placebo_intervention_metrics(policy, slice_stream, *_VELOCITY_SLICE)
    env.close()

    n_steps = metrics["n_steps"]
    argmax_flip_rate = metrics["argmax_flip_rate"]
    mean_l1_vs_zero = metrics["mean_l1_vs_zero"]
    mean_kl_vs_zero = metrics["mean_kl_vs_zero"]
    mean_l1_vs_alternatives = metrics["mean_l1_vs_alternatives"]

    print(f"Steps evaluated: {n_steps} (across {args.episodes} held-out episodes, disjoint seeds from "
          "analyze_communication.py)")
    print(f"Argmax-flip rate: {argmax_flip_rate * 100:.1f}% of steps -- fraction where forcing some *other* "
          "message than the one actually sent changes the listener's greedy movement action. Near 0% means "
          "movement doesn't causally depend on which message was sent, no matter how high MI(target; message) is.")
    by_target_rates = {t: m["argmax_flip_rate"] * 100 for t, m in sorted(by_target.items())}
    print("Argmax-flip rate by true target: " +
          ", ".join(f"target {t}={r:.1f}%" for t, r in by_target_rates.items()) +
          f" -- range={max(by_target_rates.values()) - min(by_target_rates.values()):.1f}pp "
          "(a wide range means the causal effect is heterogeneous across targets, not uniform).")
    print(f"Mean L1(real vs. zero-message) movement-distribution distance: {mean_l1_vs_zero:.4f} "
          f"(max possible: 2.0) -- Eccles et al. 2019 Eq. 8's quantity, reported as a metric here.")
    print(f"Mean KL(real || zero-message) movement distribution: {mean_kl_vs_zero:.4f} nats -- Eq. 5's CIC estimator.")
    print(f"Mean L1(real vs. every alternative message) movement-distribution distance: "
          f"{mean_l1_vs_alternatives:.4f} -- a continuous companion to argmax_flip_rate (how far the "
          "distribution moves, not just whether the argmax changes); compare directly against the placebo's "
          "mean_l1 below.")

    if placebo_supported:
        placebo_flip_rate = placebo_metrics["argmax_flip_rate"]
        placebo_mean_l1 = placebo_metrics["mean_l1"]
        print(f"Placebo (velocity-slice) flip rate: {placebo_flip_rate * 100:.1f}% of steps -- the SAME "
              "intervention procedure applied to the listener's own velocity instead of the message. If this is "
              "close to the message flip rate above, the message result mostly reflects general policy "
              "brittleness to any perturbation, not message-specific causal reliance -- see this script's "
              "docstring.")
        print(f"Message-specific signal (message flip rate minus placebo flip rate): "
              f"{(argmax_flip_rate - placebo_flip_rate) * 100:+.1f} percentage points.")
        print(f"Placebo mean L1 distance: {placebo_mean_l1:.4f} vs. message's {mean_l1_vs_alternatives:.4f} "
              f"-- message-specific action-distance signal: {mean_l1_vs_alternatives - placebo_mean_l1:+.4f}.")

    if args.json_out:
        output = {
            "n_steps": n_steps,
            "episodes": args.episodes,
            "argmax_flip_rate": argmax_flip_rate,
            "mean_l1_vs_zero": mean_l1_vs_zero,
            "mean_kl_vs_zero": mean_kl_vs_zero,
            "mean_l1_vs_alternatives": mean_l1_vs_alternatives,
            "argmax_flip_rate_by_target": by_target_rates,
        }
        if placebo_supported:
            output["placebo_argmax_flip_rate"] = placebo_metrics["argmax_flip_rate"]
            output["placebo_mean_l1"] = placebo_metrics["mean_l1"]
        with open(args.json_out, "w") as f:
            json.dump(output, f)
        print(f"Saved analysis JSON to {args.json_out}")


if __name__ == "__main__":
    main()
