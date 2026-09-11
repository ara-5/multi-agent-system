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


def make_env(num_agents: int, max_cycles: int, num_vec_envs: int):
    env = simple_spread_v3.parallel_env(
        N=num_agents, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False
    )
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3")
    return VecMonitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--out", default="models/simple_spread_ppo")
    args = parser.parse_args()

    env = make_env(args.num_agents, args.max_cycles, args.num_vec_envs)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")


if __name__ == "__main__":
    main()
