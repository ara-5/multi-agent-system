"""Record real rollout trajectories (entity positions, observations, actions) from
a trained simple_spread policy, for the in-browser live-inference demo in
web_demo/. Unlike a GIF, this dumps structured per-frame data so the browser can
render its own animation AND re-run the exact policy weights (see
export_policy_weights.py) on the recorded observations to verify it reproduces
the same actions -- real client-side neural net inference, not a canned replay.
"""
import argparse
import json

from mpe2 import simple_spread_v3
from stable_baselines3 import PPO


def entity_static_info(world):
    entities = {}
    for a in world.agents:
        entities[a.name] = {"color": a.color.tolist(), "radius": float(a.size), "kind": "agent"}
    for lm in world.landmarks:
        entities[lm.name] = {"color": lm.color.tolist(), "radius": float(lm.size), "kind": "landmark"}
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/simple_spread_ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--out", default="web_demo/data/trajectories_spread.json")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = simple_spread_v3.parallel_env(
        N=args.num_agents, local_ratio=0.5, max_cycles=args.max_cycles, continuous_actions=False,
    )

    episodes = []
    entities = None
    for ep in range(args.episodes):
        obs, _infos = env.reset(seed=args.base_seed + ep)
        world = env.unwrapped.world
        if entities is None:
            entities = entity_static_info(world)

        agent_order = list(env.agents)
        frames = [{
            "positions": {a.name: a.state.p_pos.tolist() for a in world.agents}
                       | {lm.name: lm.state.p_pos.tolist() for lm in world.landmarks},
            "obs": {a: obs[a].tolist() for a in agent_order},
            "actions": None,
            "reward": None,
        }]

        terminated = truncated = False
        while not terminated and not truncated:
            actions = {}
            for a in agent_order:
                action, _ = model.predict(obs[a], deterministic=True)
                actions[a] = int(action)
            obs, rewards, terminations, truncations, _infos = env.step(actions)
            terminated = any(terminations.values())
            truncated = any(truncations.values())
            frames.append({
                "positions": {a.name: a.state.p_pos.tolist() for a in world.agents}
                           | {lm.name: lm.state.p_pos.tolist() for lm in world.landmarks},
                "obs": {a: obs[a].tolist() for a in agent_order} if not (terminated or truncated) else None,
                "actions": actions,
                "reward": float(next(iter(rewards.values()))),
            })

        episodes.append({"seed": args.base_seed + ep, "frames": frames})

    env.close()

    with open(args.out, "w") as f:
        json.dump({"entities": entities, "agent_order": agent_order, "episodes": episodes}, f)
    print(f"Recorded {len(episodes)} episodes ({sum(len(e['frames']) for e in episodes)} frames) to {args.out}")


if __name__ == "__main__":
    main()
