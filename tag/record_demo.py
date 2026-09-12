"""Record a GIF of adversary/good policies playing simple_tag together."""
import argparse

import imageio.v2 as imageio
from mpe2 import simple_tag_v3

from common.opponent_wrapper import load_opponent_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversary-model", default=None)
    parser.add_argument("--good-model", default=None)
    parser.add_argument("--out", default="assets/demo_tag.gif")
    parser.add_argument("--num-good", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    policies = {
        "adversary": load_opponent_policy(args.adversary_model),
        "agent": load_opponent_policy(args.good_model),
    }

    env = simple_tag_v3.env(
        num_good=args.num_good, num_adversaries=args.num_adversaries, num_obstacles=args.num_obstacles,
        max_cycles=args.max_cycles, continuous_actions=False, render_mode="rgb_array",
    )

    env.reset()
    frames = [env.render()]
    for agent in env.agent_iter():
        obs, _reward, terminated, truncated, _info = env.last()
        role = "adversary" if agent.startswith("adversary_") else "agent"
        if terminated or truncated:
            action = None
        else:
            policy = policies[role]
            action = policy(agent, obs) if policy is not None else env.action_space(agent).sample()
        env.step(action)
        frames.append(env.render())
    env.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
