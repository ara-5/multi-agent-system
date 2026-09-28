"""Runs every (architecture, objective) combination this repo's design-notes
story raises but train.py/train_baseline.py don't cover individually, as one
parameterized script rather than N near-duplicate files:

--policy bottleneck   SpeakerListenerBottleneckPolicy (train.py's architecture)
--policy shared_trunk SharedTrunkAuxLossPolicy (train_baseline.py's architecture,
                       exposing the same message_logits/movement_logits_with_
                       message_override hooks the objectives below need)

--objective none      plain PPO -- train_baseline.py's original result on
                       shared_trunk; new data point on bottleneck (expected:
                       matches train.py --mi-coef 0 --ent-coef 0)
--objective entropy   generic PPO entropy bonus (--ent-coef)
--objective mi        mi_ppo.py's RIM/IMSAT channel-only MI loss (--mi-coef)
--objective eccles     eccles_ppo.py's positive-signalling + positive-listening
                       losses (--ps-coef, --pl-coef), arXiv:1912.05676

The (shared_trunk, mi) and (shared_trunk, eccles) cells are the ones that
matter most and that no existing script in this repo runs: mi_ppo.py's and
eccles_ppo.py's docstrings both predict the auxiliary loss can make the
*message* correlate with the target even without the architectural bottleneck,
because both losses only ever look at the message/movement columns of the
shared trunk's output -- not whether the *listener* is actually reading them.
If true, analyze_communication.py's MI will look identical to the bottleneck
condition, while causal_intervention.py's argmax-flip rate will reveal the
difference. Run both eval scripts on every model this trains before drawing
conclusions -- that's the whole point of this ablation.

--num-landmarks (default: unset, meaning "use the stock 3-landmark env")
switches to common/scaled_speaker_listener.py instead, for the "does any of
this hold up past a 3-symbol vocabulary" question the README's Next Steps
section raises. See that module's docstring for why this needed a real
environment (the stock env's speaker observation can't represent more than 3
distinguishable targets) rather than a training flag.
"""
import argparse
import os

import numpy as np
from bottleneck_policy import SpeakerListenerBottleneckPolicy
from eccles_ppo import EcclesPPO
from mi_diagnostic_callback import MIDiagnosticCallback
from mi_ppo import MIBonusPPO
from mpe2 import simple_speaker_listener_v4
from shared_trunk_mi_policy import SharedTrunkAuxLossPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from common.joint_env import JointPolicyEnv
from common.scaled_speaker_listener import parallel_env as scaled_parallel_env


def make_env(max_cycles: int, num_landmarks: int | None, monitor_file: str | None = None):
    if num_landmarks is None:
        pz_env_fn = lambda: simple_speaker_listener_v4.parallel_env(
            max_cycles=max_cycles, continuous_actions=False
        )
    else:
        pz_env_fn = lambda: scaled_parallel_env(
            num_landmarks=num_landmarks, max_cycles=max_cycles, continuous_actions=False
        )
    env = JointPolicyEnv(pz_env_fn)
    return Monitor(env, filename=monitor_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["bottleneck", "shared_trunk"], required=True)
    parser.add_argument("--objective", choices=["none", "entropy", "mi", "eccles"], required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-landmarks", type=int, default=None,
                         help="Unset (default): stock 3-landmark mpe2 env. Set to any value (including 3, "
                              "as a same-task sanity check -- see scaled_speaker_listener.py's docstring on "
                              "why num_landmarks=3 here isn't bit-identical to the stock env) to use "
                              "common/scaled_speaker_listener.py instead.")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--mi-coef", type=float, default=0.2)
    parser.add_argument("--ps-coef", type=float, default=0.1)
    parser.add_argument("--pl-coef", type=float, default=0.1)
    parser.add_argument("--target-entropy-frac", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--monitor-log", default=None)
    parser.add_argument("--mi-diagnostic-log", default=None,
                         help="If set, logs mi_diagnostic_callback.py's free, in-training MI estimate to this "
                              "CSV every rollout -- for every objective, not just --objective mi.")
    args = parser.parse_args()

    if args.monitor_log:
        os.makedirs(os.path.dirname(args.monitor_log), exist_ok=True)

    env = make_env(args.max_cycles, args.num_landmarks, monitor_file=args.monitor_log)

    if args.policy == "bottleneck":
        policy_class = SpeakerListenerBottleneckPolicy
        policy_kwargs = {
            "speaker_obs_dim": env.unwrapped.obs_dims[0],
            "listener_obs_dim": env.unwrapped.obs_dims[1],
            "hidden_dim": args.hidden_dim,
        }
    else:
        policy_class = SharedTrunkAuxLossPolicy
        policy_kwargs = {"speaker_obs_dim": env.unwrapped.obs_dims[0]}

    common_kwargs = {"verbose": 1, "policy_kwargs": policy_kwargs, "seed": args.seed}

    if args.objective == "none":
        model = PPO(policy_class, env, ent_coef=0.0, **common_kwargs)
    elif args.objective == "entropy":
        model = PPO(policy_class, env, ent_coef=args.ent_coef, **common_kwargs)
    elif args.objective == "mi":
        model = MIBonusPPO(policy_class, env, mi_coef=args.mi_coef, ent_coef=0.0, **common_kwargs)
    else:
        model = EcclesPPO(
            policy_class, env, ps_coef=args.ps_coef, pl_coef=args.pl_coef,
            target_entropy_frac=args.target_entropy_frac, ent_coef=0.0, **common_kwargs,
        )

    callback = None
    if args.mi_diagnostic_log:
        num_messages = env.unwrapped.action_space.nvec[0]
        callback = MIDiagnosticCallback(args.mi_diagnostic_log, max_mutual_info_bits=float(np.log2(num_messages)))

    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")


if __name__ == "__main__":
    main()
