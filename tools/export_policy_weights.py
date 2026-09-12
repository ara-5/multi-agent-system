"""Export a trained SB3 PPO policy's weights to JSON, for reimplementing the exact
same forward pass in plain JavaScript (see web_demo/). Only the policy network is
exported (not the value network) since only actions are needed for inference.

SB3's default MlpPolicy here is: FlattenExtractor (no-op on an already-flat obs
vector, no normalization) -> policy_net (Linear+Tanh, Linear+Tanh) -> action_net
(one Linear layer to logits). Action is argmax(logits) for a Discrete action
space, or per-segment argmax for MultiDiscrete (segments given by action_dims).
"""
import argparse
import json

from stable_baselines3 import PPO


def export_linear(layer):
    return {
        "W": layer.weight.detach().cpu().numpy().tolist(),  # (out_features, in_features)
        "b": layer.bias.detach().cpu().numpy().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = PPO.load(args.model)
    policy = model.policy

    hidden_layers = [
        export_linear(layer)
        for layer in policy.mlp_extractor.policy_net
        if hasattr(layer, "weight")
    ]
    action_layer = export_linear(policy.action_net)

    action_space = model.action_space
    if hasattr(action_space, "nvec"):
        action_dims = [int(n) for n in action_space.nvec]
    else:
        action_dims = [int(action_space.n)]

    data = {
        "hidden_layers": hidden_layers,
        "action_layer": action_layer,
        "action_dims": action_dims,
        "obs_dim": int(policy.observation_space.shape[0]),
    }

    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"Exported policy weights to {args.out} "
          f"(obs_dim={data['obs_dim']}, action_dims={action_dims})")


if __name__ == "__main__":
    main()
