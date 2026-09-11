"""Run a trained shared-policy model on simple_spread and report episode rewards."""
import argparse

from mpe2 import simple_spread_v3
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/simple_spread_ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = simple_spread_v3.env(
        N=args.num_agents,
        local_ratio=0.5,
        max_cycles=args.max_cycles,
        continuous_actions=False,
        render_mode="human" if args.render else None,
    )

    for episode in range(args.episodes):
        env.reset()
        total_reward = 0.0
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            total_reward += reward
            if terminated or truncated:
                action = None
            else:
                action, _ = model.predict(obs, deterministic=True)
            env.step(action)
        print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
