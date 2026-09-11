# multi-agent-system

[![CI](https://github.com/ara-5/multi-agent-system/actions/workflows/ci.yml/badge.svg)](https://github.com/ara-5/multi-agent-system/actions/workflows/ci.yml)

Multi-agent reinforcement learning (MARL) scaffold: agents that learn cooperative
behavior through training, rather than following hand-written rules.

**Stack:** [PettingZoo](https://pettingzoo.farama.org/) (multi-agent env API) +
[SuperSuit](https://github.com/Farama-Foundation/SuperSuit) (env wrappers) +
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO).

**Task:** `simple_spread` from the [MPE2](https://mpe2.farama.org/) environment suite —
N agents must cover N landmarks while avoiding collisions. Training uses parameter
sharing: one PPO policy controls every agent, vectorized across parallel env copies
via SuperSuit.

## Demo

| Random policy (untrained) | PPO policy (200k timesteps) |
| --- | --- |
| ![random policy](assets/demo_random.gif) | ![trained policy](assets/demo_trained.gif) |

![reward curve](assets/reward_curve.png)

Reward is the sum of per-agent distance-to-nearest-landmark penalties plus
collision penalties, so it's always negative — closer to zero is better. The
50-episode rolling mean improves from roughly -28 to -22 over 200k timesteps
(~9k episodes) of untuned PPO on a single CPU.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Train

```bash
python train.py --timesteps 200000
```

Saves the trained policy to `models/simple_spread_ppo.zip`. Key flags:
`--num-agents`, `--max-cycles`, `--num-vec-envs`, `--out`.

Add `--tensorboard-log runs` to `train.py` for live TensorBoard curves, or
regenerate the static reward-curve image from the CSV log:

```bash
python plot_rewards.py --log logs/monitor.csv --out assets/reward_curve.png
```

## Evaluate

```bash
python evaluate.py --model models/simple_spread_ppo --episodes 5 --render
```

Runs the trained policy for N episodes and prints total reward per episode. Drop
`--render` to run headless.

Regenerate the demo GIFs with `record_demo.py` (omit `--model` for a random-policy
baseline):

```bash
python record_demo.py --model models/simple_spread_ppo --out assets/demo_trained.gif
```

## Design notes / what I learned

- **PettingZoo's MPE environments moved to a separate `mpe2` package** in recent
  PettingZoo releases (1.27+) — most tutorials still reference the old
  `pettingzoo[mpe]` extra, which now raises an `ImportError` pointing at `mpe2`.
- **Parameter sharing (one policy, all agents) via SuperSuit's `concat_vec_envs_v1`**
  is the simplest way to get PettingZoo working with an off-the-shelf SB3
  algorithm, but it assumes homogeneous agents (same obs/action space) — it
  wouldn't work as-is for an asymmetric task like `simple_tag` (predator vs. prey
  have different roles), which would need independent per-role policies.
- **200k timesteps of untuned PPO on CPU gets modest but real improvement**
  (rolling mean reward roughly -28 → -22) — not a dramatic result, which is
  itself informative: `simple_spread`'s reward is dense and shaped, so most of
  the gain happens early, and squeezing out more would need reward-normalization,
  a learning-rate schedule, or more timesteps rather than architecture changes.
- CI runs a 4000-timestep smoke test (train + evaluate) on every push — enough to
  catch import/shape/API-breakage regressions in under a minute, without needing
  a real GPU runner or a multi-minute CI job.

## Next steps

- Try a competitive/mixed task (e.g. `simple_tag`, predator-prey) instead of the
  cooperative `simple_spread` default — this needs independent per-role policies
  since predator/prey have different observation and action spaces.
- Swap PPO for another SB3 algorithm, or compare shared vs. independent policies
  on the same task.
- Add unit tests for the deterministic parts of the pipeline (env-wrapping
  shapes, arg parsing) — the training loop itself is inherently stochastic and
  not a great unit-test target.
