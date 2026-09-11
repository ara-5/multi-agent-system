"""Record a GIF of a trained (or random, for comparison) policy on simple_spread."""
import argparse

import imageio.v2 as imageio
from mpe2 import simple_spread_v3
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to a trained model; omit for a random policy")
    parser.add_argument("--out", default="assets/demo.gif")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    model = PPO.load(args.model) if args.model else None
    env = simple_spread_v3.env(
        N=args.num_agents,
        local_ratio=0.5,
        max_cycles=args.max_cycles,
        continuous_actions=False,
        render_mode="rgb_array",
    )

    env.reset()
    frames = [env.render()]
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        elif model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        frames.append(env.render())
    env.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
