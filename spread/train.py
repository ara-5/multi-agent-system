"""Train a shared PPO policy on the PettingZoo MPE simple_spread cooperative task.

Uses SuperSuit to convert the parallel PettingZoo env into a vectorized
Stable-Baselines3 env with parameter sharing across agents (one policy
controls every agent).
"""
import argparse
import os

import supersuit as ss
from mpe2 import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor


def make_env(num_agents: int, max_cycles: int, num_vec_envs: int, monitor_file: str | None = None):
    env = simple_spread_v3.parallel_env(
        N=num_agents, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False
    )
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3")
    return VecMonitor(env, filename=monitor_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--out", default="models/simple_spread_ppo")
    parser.add_argument("--tensorboard-log", default=None, help="Directory for TensorBoard logs, e.g. runs/")
    parser.add_argument("--monitor-log", default="logs/monitor.csv", help="CSV path for per-episode reward logging")
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases (requires `wandb login` first)")
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

    env = make_env(args.num_agents, args.max_cycles, args.num_vec_envs, monitor_file=args.monitor_log)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
