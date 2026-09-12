"""Record real rollout trajectories from a trained simple_speaker_listener joint
policy, for the in-browser live-inference demo. Also records the true target
landmark and the speaker's chosen message per frame, so the demo can show
whether the message matches the target (see analyze_communication.py for the
same question, measured properly via mutual information)."""
import argparse
import json

from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO

from common.joint_env import JointPolicyEnv


def frame_positions(world):
    return ({a.name: a.state.p_pos.tolist() for a in world.agents}
            | {lm.name: lm.state.p_pos.tolist() for lm in world.landmarks})


def entity_static_info(world):
    entities = {}
    for a in world.agents:
        entities[a.name] = {"color": a.color.tolist(), "radius": float(a.size), "kind": a.name.split("_")[0]}
    for lm in world.landmarks:
        entities[lm.name] = {"color": lm.color.tolist(), "radius": float(lm.size), "kind": lm.name.replace(" ", "_")}
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/comm_joint_ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--out", default="web_demo/data/trajectories_comm.json")
    args = parser.parse_args()

    model = PPO.load(args.model)

    def make_pz_env():
        return simple_speaker_listener_v4.parallel_env(max_cycles=args.max_cycles, continuous_actions=False)

    env = JointPolicyEnv(make_pz_env)
    speaker_obs_dim = env.obs_dims[0]

    episodes = []
    entities = None
    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.base_seed + ep)
        world = env.pz_env.unwrapped.world
        if entities is None:
            entities = entity_static_info(world)

        target_index = int(max(range(speaker_obs_dim), key=lambda i: obs[i]))

        frames = [{"positions": frame_positions(world), "obs": obs.tolist(), "message": None, "reward": None}]

        terminated = truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            frames.append({
                "positions": frame_positions(world),
                "obs": obs.tolist() if not (terminated or truncated) else None,
                "message": int(action[0]),
                "reward": float(reward),
            })

        episodes.append({"seed": args.base_seed + ep, "target_index": target_index, "frames": frames})

    env.close()

    with open(args.out, "w") as f:
        json.dump({"entities": entities, "episodes": episodes}, f)
    print(f"Recorded {len(episodes)} episodes ({sum(len(e['frames']) for e in episodes)} frames) to {args.out}")


if __name__ == "__main__":
    main()
