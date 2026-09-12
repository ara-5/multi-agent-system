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
  section — the highest-value one right now is getting
  `simple_speaker_listener` to actually learn a communication protocol (see
  [Design notes](README.md#emergent-communication-design-notes) for why it
  currently doesn't, and what would likely fix it).
- A fourth PettingZoo MPE2 task added to `web_demo/` (same pattern as the
  existing three: `export_policy_weights.py` + a `record_trajectories_*.py`
  script) is a self-contained, well-scoped contribution.
- Found a bug or a claim in the README that doesn't hold up? Open an issue —
  this repo tries to report results honestly (see the `simple_speaker_listener`
  section), and a correction is welcome even if it's unflattering.
