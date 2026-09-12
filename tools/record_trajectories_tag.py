"""Record real rollout trajectories from trained simple_tag adversary/good
policies, for the in-browser live-inference demo. See record_trajectories.py for
why structured per-frame data is recorded instead of a GIF."""
import argparse
import json

from mpe2 import simple_tag_v3
from stable_baselines3 import PPO


def frame_positions(world):
    return ({a.name: a.state.p_pos.tolist() for a in world.agents}
            | {lm.name: lm.state.p_pos.tolist() for lm in world.landmarks})


def entity_static_info(world):
    entities = {}
    for a in world.agents:
        kind = "adversary" if a.adversary else "good"
        entities[a.name] = {"color": a.color.tolist(), "radius": float(a.size), "kind": kind}
    for lm in world.landmarks:
        entities[lm.name] = {"color": lm.color.tolist(), "radius": float(lm.size), "kind": "obstacle"}
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversary-model", default="models/simple_tag_adversary")
    parser.add_argument("--good-model", default="models/simple_tag_good")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-good", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--out", default="web_demo/data/trajectories_tag.json")
    args = parser.parse_args()

    models = {
        "adversary": PPO.load(args.adversary_model),
        "agent": PPO.load(args.good_model),
    }
    env = simple_tag_v3.parallel_env(
        num_good=args.num_good, num_adversaries=args.num_adversaries, num_obstacles=args.num_obstacles,
        max_cycles=args.max_cycles, continuous_actions=False,
    )

    episodes = []
    entities = None
    for ep in range(args.episodes):
        obs, _infos = env.reset(seed=args.base_seed + ep)
        world = env.unwrapped.world
        if entities is None:
            entities = entity_static_info(world)

        agent_order = list(env.agents)

        frames = [{"positions": frame_positions(world), "obs": {a: obs[a].tolist() for a in agent_order},
                   "actions": None, "reward": None}]

        terminated = truncated = False
        while not terminated and not truncated:
            actions = {}
            for a in agent_order:
                role = "adversary" if a.startswith("adversary_") else "agent"
                action, _ = models[role].predict(obs[a], deterministic=True)
                actions[a] = int(action)
            obs, rewards, terminations, truncations, _infos = env.step(actions)
            terminated = any(terminations.values())
            truncated = any(truncations.values())
            frames.append({
                "positions": frame_positions(world),
                "obs": {a: obs[a].tolist() for a in agent_order} if not (terminated or truncated) else None,
                "actions": actions,
                "reward": {a: float(rewards[a]) for a in agent_order},
            })

        episodes.append({"seed": args.base_seed + ep, "frames": frames})

    env.close()

    with open(args.out, "w") as f:
        json.dump({"entities": entities, "agent_order": agent_order, "episodes": episodes}, f)
    print(f"Recorded {len(episodes)} episodes ({sum(len(e['frames']) for e in episodes)} frames) to {args.out}")


if __name__ == "__main__":
    main()
