# multi-agent-system

Multi-agent reinforcement learning (MARL) scaffold: agents that learn cooperative
behavior through training, rather than following hand-written rules.

**Stack:** [PettingZoo](https://pettingzoo.farama.org/) (multi-agent env API) +
[SuperSuit](https://github.com/Farama-Foundation/SuperSuit) (env wrappers) +
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO).

**Task:** `simple_spread` from the [MPE2](https://mpe2.farama.org/) environment suite —
N agents must cover N landmarks while avoiding collisions. Training uses parameter
sharing: one PPO policy controls every agent, vectorized across parallel env copies
via SuperSuit.

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

## Evaluate

```bash
python evaluate.py --model models/simple_spread_ppo --episodes 5 --render
```

Runs the trained policy for N episodes and prints total reward per episode. Drop
`--render` to run headless.

## Next steps

- Try a competitive/mixed task (e.g. `simple_tag`, predator-prey) instead of the
  cooperative `simple_spread` default.
- Swap PPO for another SB3 algorithm, or move to independent (non-shared) policies
  per agent.
- Log training curves with `--tensorboard-log` (SB3 supports this out of the box)
  for a real learning-curve view instead of just stdout.
