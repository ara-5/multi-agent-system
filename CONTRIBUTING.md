# Contributing

Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check .
pytest tests/
```

Both run in CI on every push, along with tiny end-to-end smoke tests for all
three tasks (train + evaluate at a few thousand timesteps) — keep new training
scripts consistent with that pattern so CI can smoke-test them too.

## Where to start

- Open items are tracked in the README's [Next steps](README.md#next-steps)
  section — the highest-value one right now is pushing
  `simple_speaker_listener` from a partial (13.5% of max mutual information)
  to a clean communication protocol (see
  [Design notes](README.md#emergent-communication-design-notes) for what's
  been tried and what would likely close the rest of the gap).
- A fourth PettingZoo MPE2 task, following the existing `common/` +
  `<task>/{train,evaluate,record_demo}.py` pattern, plus a
  `tools/export_policy_weights.py` + `tools/record_trajectories_*.py` pair to
  wire it into `web_demo/`, is a self-contained, well-scoped contribution.
- Found a bug or a claim in the README that doesn't hold up? Open an issue —
  this repo tries to report results honestly (see the `simple_speaker_listener`
  section), and a correction is welcome even if it's unflattering.
