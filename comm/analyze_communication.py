"""Check whether the trained joint policy actually learned a communication
protocol: for many episodes, record the true target landmark (from the
speaker's own one-hot-ish observation) against the discrete message the speaker
sends at step 0, and tabulate target vs. message as a confusion matrix. A
diagonal-ish matrix means a consistent (if arbitrary) protocol emerged; a
uniform/scattered matrix means the message carries no information.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO

from common.joint_env import JointPolicyEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/comm_joint_ppo")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", default="assets/comm_confusion_matrix.png")
    parser.add_argument("--json-out", default=None,
                         help="Optional path to also dump {confusion, mutual_info_bits, "
                              "max_mutual_info_bits, episodes} as JSON")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = JointPolicyEnv(lambda: simple_speaker_listener_v4.parallel_env(
        max_cycles=25, continuous_actions=False,
    ))

    speaker_obs_dim = env.obs_dims[0]
    num_targets = speaker_obs_dim
    num_messages = env.action_space.nvec[0]
    confusion = np.zeros((num_targets, num_messages), dtype=int)

    for episode in range(args.episodes):
        obs, _info = env.reset(seed=args.base_seed + episode)
        target = int(np.argmax(obs[:speaker_obs_dim]))
        action, _ = model.predict(obs, deterministic=True)
        message = int(action[0])
        confusion[target, message] += 1

    env.close()

    print("Target (row) vs. message (col) counts:")
    print(confusion)

    p_joint = confusion / confusion.sum()
    p_target = p_joint.sum(axis=1, keepdims=True)
    p_message = p_joint.sum(axis=0, keepdims=True)
    p_independent = p_target * p_message
    mutual_info = np.sum(p_joint * np.log2((p_joint + 1e-12) / (p_independent + 1e-12)))
    max_mutual_info = np.log2(num_messages)
    print(f"Mutual information between target and message: {mutual_info:.4f} bits "
          f"({mutual_info / max_mutual_info * 100:.1f}% of the {max_mutual_info:.4f}-bit max) "
          "-- near 0% means the message carries essentially no information about the target.")

    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.xlabel("Speaker message")
    plt.ylabel("True target landmark")
    plt.xticks(range(num_messages))
    plt.yticks(range(num_targets))
    plt.title(f"Learned protocol ({args.episodes} eps, "
              f"{mutual_info / max_mutual_info * 100:.0f}% of max mutual info)")
    for i in range(num_targets):
        for j in range(num_messages):
            plt.text(j, i, confusion[i, j], ha="center", va="center")
    plt.colorbar(label="episode count")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved confusion matrix to {args.out}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "confusion": confusion.tolist(),
                "mutual_info_bits": float(mutual_info),
                "max_mutual_info_bits": float(max_mutual_info),
                "episodes": args.episodes,
            }, f)
        print(f"Saved analysis JSON to {args.json_out}")


if __name__ == "__main__":
    main()
