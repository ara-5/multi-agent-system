"""Record a GIF of a trained (or random, for comparison) joint policy playing
simple_speaker_listener."""
import argparse

import imageio.v2 as imageio
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO

from joint_env import JointPolicyEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to a trained model; omit for a random policy")
    parser.add_argument("--out", default="assets/demo_comm.gif")
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    model = PPO.load(args.model) if args.model else None
    env = JointPolicyEnv(lambda: simple_speaker_listener_v4.parallel_env(
        max_cycles=args.max_cycles, continuous_actions=False, render_mode="rgb_array",
    ))

    obs, _info = env.reset()
    frames = [env.render()]
    terminated = truncated = False
    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _reward, terminated, truncated, _info = env.step(action)
        frames.append(env.render())
    env.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
