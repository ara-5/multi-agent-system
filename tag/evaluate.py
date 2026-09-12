"""Run adversary (predator) and good (prey) policies together on simple_tag and
report per-group episode rewards. Omit either --adversary-model or --good-model to
use a random policy for that side (e.g. to sanity-check a trained side against a
random baseline opponent).

With --episodes > 1 and no --render, each episode uses a different seed
(--base-seed + episode index) and a mean +/- std summary is printed at the end.
"""
import argparse
from collections import defaultdict

import numpy as np
from mpe2 import simple_tag_v3

from common.opponent_wrapper import load_opponent_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversary-model", default=None)
    parser.add_argument("--good-model", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-good", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    policies = {
        "adversary": load_opponent_policy(args.adversary_model),
        "agent": load_opponent_policy(args.good_model),
    }

    env = simple_tag_v3.env(
        num_good=args.num_good, num_adversaries=args.num_adversaries, num_obstacles=args.num_obstacles,
        max_cycles=args.max_cycles, continuous_actions=False,
        render_mode="human" if args.render else None,
    )

    adversary_rewards, prey_rewards = [], []
    for episode in range(args.episodes):
        env.reset(seed=args.base_seed + episode)
        totals = defaultdict(float)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _info = env.last()
            role = "adversary" if agent.startswith("adversary_") else "agent"
            totals[role] += reward
            if terminated or truncated:
                action = None
            else:
                policy = policies[role]
                action = policy(agent, obs) if policy is not None else env.action_space(agent).sample()
            env.step(action)
        adversary_rewards.append(totals["adversary"])
        prey_rewards.append(totals["agent"])
        print(f"Episode {episode + 1}: adversaries total = {totals['adversary']:.2f}, "
              f"prey total = {totals['agent']:.2f}")

    env.close()

    adversary_rewards, prey_rewards = np.array(adversary_rewards), np.array(prey_rewards)
    print(f"\nMean over {args.episodes} episodes: "
          f"adversaries = {adversary_rewards.mean():.2f} +/- {adversary_rewards.std():.2f}, "
          f"prey = {prey_rewards.mean():.2f} +/- {prey_rewards.std():.2f}")


if __name__ == "__main__":
    main()
