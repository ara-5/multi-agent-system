"""Train a joint PPO policy on simple_speaker_listener: a speaker that can see the
target landmark but can't move, and a listener that can move but can't see the
target -- they must develop a shared communication protocol (the speaker's
discrete "utterance") for the listener to reach the right landmark. See
joint_env.py for why this uses a single joint policy rather than the
freeze-one-side approach used for simple_tag.
"""
import argparse
import os

from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from joint_env import JointPolicyEnv


def make_env(max_cycles: int, monitor_file: str | None = None):
    env = JointPolicyEnv(lambda: simple_speaker_listener_v4.parallel_env(
        max_cycles=max_cycles, continuous_actions=False
    ))
    return Monitor(env, filename=monitor_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--ent-coef", type=float, default=0.02,
                         help="PPO entropy coefficient; >0 discourages premature convergence to a "
                              "low-entropy (e.g. channel-ignoring) policy. 200k steps at the default "
                              "0.0 gets ~0.9%% of max mutual information between target and message; "
                              "1M steps at 0.02 gets ~13.5%% -- see Design notes in the README.")
    parser.add_argument("--out", default="models/comm_joint_ppo")
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
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=args.ent_coef, tensorboard_log=args.tensorboard_log)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
