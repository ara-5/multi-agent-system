"""Run a trained joint policy on simple_speaker_listener and report episode rewards.

With --episodes > 1, each episode uses a different seed (--base-seed + episode
index) and a mean +/- std summary is printed at the end.
"""
import argparse

import numpy as np
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO

from common.joint_env import JointPolicyEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/comm_joint_ppo")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=25)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = JointPolicyEnv(lambda: simple_speaker_listener_v4.parallel_env(
        max_cycles=args.max_cycles, continuous_actions=False
    ))

    episode_rewards = []
    for episode in range(args.episodes):
        obs, _info = env.reset(seed=args.base_seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

    env.close()

    rewards = np.array(episode_rewards)
    print(f"\nMean reward over {args.episodes} episodes: {rewards.mean():.2f} +/- {rewards.std():.2f}")


if __name__ == "__main__":
    main()
