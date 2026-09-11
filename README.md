# multi-agent-system

[![CI](https://github.com/ara-5/multi-agent-system/actions/workflows/ci.yml/badge.svg)](https://github.com/ara-5/multi-agent-system/actions/workflows/ci.yml)

Multi-agent reinforcement learning (MARL) scaffold: agents that learn cooperative
*and* competitive behavior through training, rather than following hand-written
rules.

**Stack:** [PettingZoo](https://pettingzoo.farama.org/) (multi-agent env API) +
[SuperSuit](https://github.com/Farama-Foundation/SuperSuit) (env wrappers) +
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO).

**Tasks**, both from the [MPE2](https://mpe2.farama.org/) environment suite:

- **`simple_spread`** (cooperative) — N agents must cover N landmarks while
  avoiding collisions. Trained via parameter sharing: one PPO policy controls
  every agent, since all agents are homogeneous (same obs/action space).
- **`simple_tag`** (competitive, predator/prey) — adversaries chase good agents.
  The two roles have different observation spaces and opposing rewards, so
  parameter sharing doesn't apply; see [Independent policies](#independent-policies-simple_tag)
  below for how this repo trains genuinely separate per-role policies.

## Demo: simple_spread (cooperative)

| Random policy (untrained) | PPO policy (200k timesteps) |
| --- | --- |
| ![random policy](assets/demo_random.gif) | ![trained policy](assets/demo_trained.gif) |

![reward curve](assets/reward_curve.png)

Reward is the sum of per-agent distance-to-nearest-landmark penalties plus
collision penalties, so it's always negative — closer to zero is better. The
50-episode rolling mean improves from roughly -28 to -22 over 200k timesteps
(~9k episodes) of untuned PPO on a single CPU.

## Demo: simple_tag (competitive, predator/prey)

![trained predator vs. trained prey](assets/demo_tag.gif)

| Adversary (predator) training | Prey training (vs. the trained adversary above) |
| --- | --- |
| ![adversary reward curve](assets/tag_adversary_reward_curve.png) | ![prey reward curve](assets/tag_good_reward_curve.png) |

### Results (20 seeded episodes each, mean ± std total reward)

| Adversary policy | Prey policy | Adversary reward | Prey reward |
| --- | --- | --- | --- |
| trained | trained | 33.00 ± 42.32 | -11.60 ± 13.82 |
| trained | random | 28.50 ± 47.88 | -23.13 ± 22.45 |
| random | trained | 15.00 ± 32.17 | -5.93 ± 10.66 |
| random | random | 10.50 ± 27.29 | -14.57 ± 17.05 |

Both trained policies clearly beat their random counterparts regardless of
opponent: the trained adversary nearly 3x's random-adversary reward (28.5-33 vs.
10.5-15), and trained prey roughly halves its penalty versus random prey (-5.93 to
-11.60 vs. -14.57 to -23.13). High variance is expected — a handful of adversaries
either catch prey (discrete +10/-10 collision reward) or don't in a given 25-step
episode, so per-episode reward is closer to a coin flip than a smooth signal;
that's why the table reports 20-episode means rather than a single run.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e .
```

For experiment tracking on [Weights & Biases](https://wandb.ai/) instead of static
images, install the optional extra and log in once (`wandb login`), then pass
`--wandb` to `train.py`:

```bash
pip install -e ".[wandb]"
python train.py --wandb --wandb-project multi-agent-system
```

## Train: simple_spread

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

## Train: simple_tag <a name="independent-policies-simple_tag"></a>

Stable-Baselines3 trains one policy per env, but `simple_tag`'s two roles
(adversary/predator vs. good/prey) have different observation spaces and opposing
rewards — parameter sharing across them isn't meaningful. This repo instead uses
`opponent_wrapper.py`'s `FixedOpponentWrapper` to expose only one role as
controllable per training run, with the other role acting via a fixed policy
(random, or a previously trained model): train adversaries vs. a random prey
baseline, then train prey against the now-frozen trained adversary.

```bash
python train_tag.py --role adversary --timesteps 100000 --out models/simple_tag_adversary
python train_tag.py --role good --opponent-model models/simple_tag_adversary \
    --timesteps 100000 --out models/simple_tag_good
```

This is **not** true simultaneous self-play (both sides improving together, which
would need a multi-policy trainer like RLlib) — it's a simpler two-stage
alternative that still produces genuinely independent, role-specific policies.

## Evaluate

```bash
python evaluate.py --model models/simple_spread_ppo --episodes 20
python evaluate_tag.py --adversary-model models/simple_tag_adversary \
    --good-model models/simple_tag_good --episodes 20
```

Runs N seeded episodes and prints a mean ± std reward summary — a single episode
is too noisy to be a meaningful result on its own (see the simple_tag results
table above). Add `--render` to watch one episode interactively instead. Omit
either `--adversary-model`/`--good-model` to use a random policy for that role,
useful for baseline comparisons.

Regenerate the demo GIFs (omit `--model`/`--*-model` args for a random-policy
baseline):

```bash
python record_demo.py --model models/simple_spread_ppo --out assets/demo_trained.gif
python record_demo_tag.py --adversary-model models/simple_tag_adversary \
    --good-model models/simple_tag_good --out assets/demo_tag.gif
```

## Test

```bash
pip install -e ".[dev]"
ruff check .
pytest tests/
```

Unit tests cover `FixedOpponentWrapper`'s agent-filtering/action-merging logic
against a fake env — the training loops themselves are stochastic and not a
useful unit-test target, so CI instead smoke-tests them end-to-end (tiny
timestep counts, real environments) rather than trying to assert on outcomes.

## Design notes / what I learned

- **PettingZoo's MPE environments moved to a separate `mpe2` package** in recent
  PettingZoo releases (1.27+) — most tutorials still reference the old
  `pettingzoo[mpe]` extra, which now raises an `ImportError` pointing at `mpe2`.
- **Parameter sharing (one policy, all agents) via SuperSuit's `concat_vec_envs_v1`**
  is the simplest way to get PettingZoo working with an off-the-shelf SB3
  algorithm, but it assumes homogeneous agents (same obs/action space) — it
  doesn't work for an asymmetric task like `simple_tag`, which is why that task
  uses `FixedOpponentWrapper` instead (see [Train: simple_tag](#independent-policies-simple_tag)).
- **200k timesteps of untuned PPO on CPU gets modest but real improvement**
  (rolling mean reward roughly -28 → -22) — not a dramatic result, which is
  itself informative: `simple_spread`'s reward is dense and shaped, so most of
  the gain happens early, and squeezing out more would need reward-normalization,
  a learning-rate schedule, or more timesteps rather than architecture changes.
- **The training-curve and evaluation-time reward numbers use different
  accounting, on purpose** — don't be alarmed that they don't match. During
  training, SuperSuit vectorizes each agent into its own env slot, so
  `VecMonitor`'s per-episode reward in `assets/reward_curve.png` is a *single
  agent's* return. `evaluate.py` instead sums reward across every agent's turn
  in the AEC loop; since `simple_spread` gives every agent the same shared team
  reward each step, that sum is roughly `num_agents`x larger in magnitude
  (hence -22 during training vs. -66 in a 3-agent evaluation of the same policy).
- **`simple_tag`'s reward is sparse and discrete** (±10 per collision, decided in
  a 25-step episode) rather than `simple_spread`'s dense distance shaping — that's
  why its evaluation numbers have much higher variance and need a 20-episode mean
  to say anything, versus a single episode being roughly informative for
  `simple_spread`.
- CI runs tiny smoke tests (train + evaluate, both tasks) plus the real unit
  test suite on every push — enough to catch import/shape/API-breakage
  regressions in under a couple of minutes, without needing a real GPU runner.

## Next steps

- **Emergent communication** (planned next) — PettingZoo's
  `simple_speaker_listener`/`simple_reference` environments, where one agent
  can observe the goal but can't act on it and another can act but can't
  observe it, forcing a learned communication protocol between them.
- Swap PPO for another SB3 algorithm, or attempt true simultaneous self-play
  (both `simple_tag` roles improving together, e.g. via RLlib) instead of the
  freeze-one-side approach used here.
- A natural-language "mission control" layer on trained policies (via the
  Claude API) as a stretch/demo feature, once the above lands.
