"""Train simple_speaker_listener with an architectural information bottleneck
that forces a real communication protocol to emerge -- see bottleneck_policy.py
for why. train_baseline.py is the earlier, naive single-shared-trunk approach,
kept for the record: it plateaus at 13.5% of max mutual information between
target and message no matter how it's tuned, because nothing in that
architecture ever requires the message to carry the information at all.

The bottleneck architecture alone (--ent-coef 0.0 --mi-coef 0.0) reliably
reaches ~55-60% of max mutual information across seeds -- a stable but
partial protocol using only 2 of the 3 message symbols. A generic entropy
bonus (--ent-coef > 0) was tried as a fix for that and turned out to be a
high-variance gamble across seeds (99.7%, 0.0%, 55.3% mutual information in a
3-seed rerun -- see Design notes in the README): untargeted, it destabilizes
the *whole* joint action distribution, not just the message channel.

--mi-coef (now the default, at 0.2) is a different, targeted fix:
mi_ppo.py's MIBonusPPO adds an auxiliary loss (the RIM/IMSAT
mutual-information-maximization regularizer) computed only from the
speaker's message logits, pushing the message to be both confident given the
input and diverse across the batch. Unlike the entropy bonus, this reached
the full clean protocol (99.7% of max MI) on 3/3 random seeds, not 1/3 --
see Design notes for the full comparison and mi_ppo.py for the derivation.
"""
import argparse
import os

from bottleneck_policy import SpeakerListenerBottleneckPolicy
from mi_ppo import MIBonusPPO
from mpe2 import simple_speaker_listener_v4
from stable_baselines3.common.monitor import Monitor

from common.joint_env import JointPolicyEnv


def make_env(max_cycles: int, monitor_file: str | None = None):
    env = JointPolicyEnv(lambda: simple_speaker_listener_v4.parallel_env(
        max_cycles=max_cycles, continuous_actions=False
    ))
    return Monitor(env, filename=monitor_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--ent-coef", type=float, default=0.0,
                         help="Generic PPO entropy bonus over the whole joint action. Found to be an "
                              "unreliable fix for the 2-of-3-message local optimum (high seed variance) "
                              "-- see Design notes in the README. Defaults to off; --mi-coef is the "
                              "more targeted alternative.")
    parser.add_argument("--mi-coef", type=float, default=0.2,
                         help="Coefficient for MIBonusPPO's message-only mutual-information auxiliary "
                              "loss (see mi_ppo.py). 0.0 makes this identical to plain PPO. The default "
                              "0.2 reached 99.7%% of max mutual information on 3/3 random seeds -- see "
                              "Design notes in the README.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None,
                         help="Seeds network init, env resets during training, and action sampling "
                              "(via SB3's set_random_seed) -- for multi-seed reproducibility checks.")
    parser.add_argument("--out", default="models/comm_ppo")
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--monitor-log", default="logs/comm_monitor.csv")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="multi-agent-system")
    args = parser.parse_args()

    if args.monitor_log:
        os.makedirs(os.path.dirname(args.monitor_log), exist_ok=True)

    run = None
    callback = None
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        args.tensorboard_log = args.tensorboard_log or "runs"
        run = wandb.init(project=args.wandb_project, config=vars(args), sync_tensorboard=True)
        callback = WandbCallback(verbose=2)

    env = make_env(args.max_cycles, monitor_file=args.monitor_log)
    policy_kwargs = {
        "speaker_obs_dim": env.unwrapped.obs_dims[0],
        "listener_obs_dim": env.unwrapped.obs_dims[1],
        "hidden_dim": args.hidden_dim,
    }
    model = MIBonusPPO(
        SpeakerListenerBottleneckPolicy, env, verbose=1, ent_coef=args.ent_coef, mi_coef=args.mi_coef,
        policy_kwargs=policy_kwargs, tensorboard_log=args.tensorboard_log, seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
