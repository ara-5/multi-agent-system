"""Communication-noise robustness sweep: does the learned protocol degrade
gracefully as the transmitted message is corrupted, or does it collapse?

Every other analysis in this repo evaluates the channel exactly as trained --
clean transmission. That answers "did a protocol emerge?" but not "is it a
protocol a real, unreliable channel could survive?" This script answers the
second question by independently corrupting the speaker's transmitted message
with probability p at every step (replacing it with a uniformly random
*different* message before the environment advances), for p in a swept range
of noise levels, and measuring how task reward degrades.

Corrupting at the environment-step level, not just the observation, matters:
the listener's next observation's communication slice comes from whatever
discrete action the speaker actually took that step, so replacing the
speaker's action before env.step() is exactly what a lossy channel would do
to the transmitted symbol -- nothing else about the world (positions,
velocities, reward) is touched.

Two things are measured at each noise level:
  - reward_mean / reward_std: standard episode-reward degradation curve.
  - decision_flip_rate: of the steps a corruption event actually happened on,
    the fraction where the listener's greedy movement action -- recomputed
    via movement_logits_with_message_override, the same hook
    causal_intervention.py uses -- actually differs between "message as
    intended" and "message as corrupted." This is the redundancy question:
    a protocol using more of its message space non-uniformly, or relying on
    finer distinctions, will have a *higher* flip rate (less redundant, more
    fragile) even if reward degrades similarly, since reward only reflects
    the physical consequence, not how close a call the corruption was.
"""
import argparse
import json

import numpy as np
import torch as th
from mpe2 import simple_speaker_listener_v4
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from torch.nn import functional as F

from common.joint_env import JointPolicyEnv
from common.scaled_speaker_listener import parallel_env as scaled_parallel_env


def _run_noise_level(env, model, num_messages, noise_p, episodes, base_seed, corruption_rng,
                      track_decision_flip):
    policy = model.policy
    episode_rewards = []
    corrupted_steps = 0
    decision_flips = 0

    for episode in range(episodes):
        obs, _info = env.reset(seed=base_seed + episode)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            intended_message = int(action[0])

            if corruption_rng.random() < noise_p:
                alternatives = [m for m in range(num_messages) if m != intended_message]
                sent_message = int(corruption_rng.choice(alternatives))
                if track_decision_flip:
                    obs_tensor = obs_as_tensor(np.expand_dims(obs, 0), policy.device)
                    features = policy.extract_features(obs_tensor)
                    with th.no_grad():
                        intended_logits = policy.movement_logits_with_message_override(
                            features, F.one_hot(th.tensor([intended_message]), num_messages).float()
                        )
                        sent_logits = policy.movement_logits_with_message_override(
                            features, F.one_hot(th.tensor([sent_message]), num_messages).float()
                        )
                    if int(intended_logits.argmax(dim=1).item()) != int(sent_logits.argmax(dim=1).item()):
                        decision_flips += 1
                corrupted_steps += 1
                action = action.copy()
                action[0] = sent_message

            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)

    rewards = np.array(episode_rewards)
    return {
        "noise_pct": noise_p * 100,
        "episodes": episodes,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "corrupted_steps": corrupted_steps,
        "decision_flip_rate": (decision_flips / corrupted_steps) if corrupted_steps > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--noise-levels", default="0,0.05,0.1,0.2,0.3",
                         help="Comma-separated per-step message-corruption probabilities.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20_000,
                         help="Disjoint from analyze_communication.py's 0 and causal_intervention.py's 10_000, "
                              "so this script's episodes are genuinely held-out from both.")
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--num-landmarks", type=int, default=None,
                         help="Must match what the model was trained with -- see train_ablation.py.")
    parser.add_argument("--corruption-seed", type=int, default=0,
                         help="Seeds the RNG deciding which steps get corrupted and which alternate message is "
                              "substituted -- independent of --base-seed's environment/episode seeding.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    model = PPO.load(args.model)
    policy = model.policy
    track_decision_flip = hasattr(policy, "movement_logits_with_message_override")
    if not track_decision_flip:
        print(f"Warning: {type(policy).__name__} doesn't expose movement_logits_with_message_override -- "
              "only reward degradation will be measured, not decision-flip rate.")

    if args.num_landmarks is None:
        pz_env_fn = lambda: simple_speaker_listener_v4.parallel_env(
            max_cycles=args.max_cycles, continuous_actions=False,
        )
    else:
        pz_env_fn = lambda: scaled_parallel_env(
            num_landmarks=args.num_landmarks, max_cycles=args.max_cycles, continuous_actions=False,
        )
    env = JointPolicyEnv(pz_env_fn)
    num_messages = int(env.action_space.nvec[0])

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    corruption_rng = np.random.default_rng(args.corruption_seed)

    results = []
    for noise_p in noise_levels:
        result = _run_noise_level(
            env, model, num_messages, noise_p, args.episodes, args.base_seed, corruption_rng,
            track_decision_flip,
        )
        results.append(result)
        flip_str = (f"{result['decision_flip_rate'] * 100:.1f}%" if result["decision_flip_rate"] is not None
                    else "n/a")
        print(f"noise={noise_p * 100:5.1f}%  reward={result['reward_mean']:7.2f} +/- {result['reward_std']:.2f}  "
              f"corrupted_steps={result['corrupted_steps']:5d}  decision_flip_rate={flip_str}")

    env.close()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f)
        print(f"Saved noise-sweep JSON to {args.json_out}")


if __name__ == "__main__":
    main()
