"""Export a trained SB3 PPO policy's weights to JSON, for reimplementing the exact
same forward pass in plain JavaScript (see web_demo/). Only the policy network is
exported (not the value network) since only actions are needed for inference.

Handles two architectures:
- SB3's default MlpPolicy: FlattenExtractor (no-op on an already-flat obs
  vector, no normalization) -> policy_net (Linear+Tanh, Linear+Tanh) ->
  action_net (one Linear layer to logits). Action is argmax(logits) for a
  Discrete action space, or per-segment argmax for MultiDiscrete (segments
  given by action_dims).
- comm/bottleneck_policy.py's SpeakerListenerBottleneckPolicy: two independent
  sub-networks (detected via the mlp_extractor's speaker_net attribute), each
  Linear+Tanh, Linear+Tanh -> its own head, with no shared layer between them.
  Exported as {"type": "speaker_listener_bottleneck", "speaker_dim", ...}
  instead, since it's not one trunk + one action layer.
"""
import argparse
import json
import os
import sys

# A model trained via `python comm/train.py` pickles its custom policy class
# under the bare module name "bottleneck_policy" (comm/'s own directory was
# sys.path[0] at save time, not the repo root) -- so PPO.load needs that same
# bare name importable here too, regardless of where this script is run from.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comm"))

from stable_baselines3 import PPO


def export_linear(layer):
    return {
        "W": layer.weight.detach().cpu().numpy().tolist(),  # (out_features, in_features)
        "b": layer.bias.detach().cpu().numpy().tolist(),
    }


def export_default_mlp_policy(policy, action_space):
    hidden_layers = [
        export_linear(layer)
        for layer in policy.mlp_extractor.policy_net
        if hasattr(layer, "weight")
    ]
    action_layer = export_linear(policy.action_net)

    if hasattr(action_space, "nvec"):
        action_dims = [int(n) for n in action_space.nvec]
    else:
        action_dims = [int(action_space.n)]

    return {
        "type": "mlp",
        "hidden_layers": hidden_layers,
        "action_layer": action_layer,
        "action_dims": action_dims,
        "obs_dim": int(policy.observation_space.shape[0]),
    }


def export_speaker_listener_bottleneck_policy(policy):
    extractor = policy.mlp_extractor
    return {
        "type": "speaker_listener_bottleneck",
        "speaker_dim": extractor.speaker_dim,
        "listener_dim": extractor.listener_dim,
        "speaker_hidden_layers": [export_linear(layer) for layer in extractor.speaker_net if hasattr(layer, "weight")],
        "message_head": export_linear(extractor.message_head),
        "listener_hidden_layers": [export_linear(layer) for layer in extractor.listener_net if hasattr(layer, "weight")],
        "movement_head": export_linear(extractor.movement_head),
        "obs_dim": int(policy.observation_space.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = PPO.load(args.model)
    policy = model.policy

    if hasattr(policy.mlp_extractor, "speaker_net"):
        data = export_speaker_listener_bottleneck_policy(policy)
    else:
        data = export_default_mlp_policy(policy, model.action_space)

    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"Exported policy weights to {args.out} (type={data['type']}, obs_dim={data['obs_dim']})")


if __name__ == "__main__":
    main()
