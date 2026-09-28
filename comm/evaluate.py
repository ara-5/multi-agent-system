"""Run a trained joint policy on simple_speaker_listener and report episode rewards.

With --episodes > 1, each episode uses a different seed (--base-seed + episode
index) and a mean +/- std summary is printed at the end.
"""
import argparse

import numpy as np
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO

from common.joint_env import JointPolicyEnv
from common.scaled_speaker_listener import parallel_env as scaled_parallel_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/comm_ppo",
                         help="Path to a trained model; pass an empty string for a random-policy baseline")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-landmarks", type=int, default=None,
                         help="Must match what the model was trained with -- see train_ablation.py.")
    args = parser.parse_args()

    model = PPO.load(args.model) if args.model else None
    if args.num_landmarks is None:
        pz_env_fn = lambda: simple_speaker_listener_v4.parallel_env(
            max_cycles=args.max_cycles, continuous_actions=False
        )
    else:
        pz_env_fn = lambda: scaled_parallel_env(
            num_landmarks=args.num_landmarks, max_cycles=args.max_cycles, continuous_actions=False
        )
    env = JointPolicyEnv(pz_env_fn)

    episode_rewards = []
    for episode in range(args.episodes):
        obs, _info = env.reset(seed=args.base_seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

    env.close()

    rewards = np.array(episode_rewards)
    print(f"\nMean reward over {args.episodes} episodes: {rewards.mean():.2f} +/- {rewards.std():.2f}")


if __name__ == "__main__":
    main()
