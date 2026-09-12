"""Train simple_speaker_listener with an architectural information bottleneck
that forces a real communication protocol to emerge -- see bottleneck_policy.py
for why. train_baseline.py is the earlier, naive single-shared-trunk approach,
kept for the record: it plateaus at 13.5% of max mutual information between
target and message no matter how it's tuned, because nothing in that
architecture ever requires the message to carry the information at all. This
script fixes that at the root and reaches 99.7% (see Design notes in the
README for the full before/after story and why the entropy bonus below is
still needed on top of the architecture fix).
"""
import argparse
import os

from bottleneck_policy import SpeakerListenerBottleneckPolicy
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO
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
    parser.add_argument("--ent-coef", type=float, default=0.01,
                         help="The bottleneck architecture alone (ent_coef=0.0) already forces the "
                              "message to carry real target information -- 300k steps gets 55.3%% of "
                              "max mutual information -- but it can settle for a deterministic protocol "
                              "that only uses 2 of 3 message symbols (distinguishing one target from "
                              "the other two, but not those two from each other), since that's already "
                              "enough for decent reward. A small entropy bonus discourages settling for "
                              "that partial equilibrium and gets 99.7%% -- see Design notes in the README.")
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
    model = PPO(
        SpeakerListenerBottleneckPolicy, env, verbose=1, ent_coef=args.ent_coef,
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
