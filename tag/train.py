"""Train an independent PPO policy for one role in the MPE2 simple_tag competitive
task (predators = "adversary", prey = "good"), with the opposing role played by a
fixed policy via FixedOpponentWrapper. See opponent_wrapper.py for why this
freeze-one-side approach is used instead of true simultaneous self-play.
"""
import argparse
import os

import supersuit as ss
from mpe2 import simple_tag_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from common.opponent_wrapper import FixedOpponentWrapper, load_opponent_policy


def make_env(role, num_good, num_adversaries, num_obstacles, max_cycles, num_vec_envs, opponent_model, monitor_file=None):
    prefix = "adversary_" if role == "adversary" else "agent_"
    opponent_policy = load_opponent_policy(opponent_model)

    env = simple_tag_v3.parallel_env(
        num_good=num_good, num_adversaries=num_adversaries, num_obstacles=num_obstacles,
        max_cycles=max_cycles, continuous_actions=False,
    )
    env = FixedOpponentWrapper(env, controlled_prefix=prefix, opponent_policy=opponent_policy)
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3")
    return VecMonitor(env, filename=monitor_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["adversary", "good"], required=True,
                         help="Which side to train: 'adversary' (predators) or 'good' (prey)")
    parser.add_argument("--opponent-model", default=None,
                         help="Path to a trained model for the other side; omit for a random opponent")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--num-good", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--monitor-log", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="multi-agent-system")
    args = parser.parse_args()

    args.out = args.out or f"models/simple_tag_{args.role}"
    args.monitor_log = args.monitor_log or f"logs/tag_{args.role}_monitor.csv"
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

    env = make_env(
        args.role, args.num_good, args.num_adversaries, args.num_obstacles,
        args.max_cycles, args.num_vec_envs, args.opponent_model, monitor_file=args.monitor_log,
    )
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
