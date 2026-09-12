"""Unit tests for FixedOpponentWrapper's agent-filtering/action-merging logic.

Uses a minimal fake ParallelEnv instead of a real PettingZoo environment, since
the wrapper's behavior is pure bookkeeping that doesn't depend on any particular
env's dynamics -- these should run in milliseconds with no training involved.
"""
from common.opponent_wrapper import FixedOpponentWrapper


class FakeSpace:
    """A stand-in for a gymnasium Space: supports both `.sample()` (used by the
    wrapper) and equality against the string used in the observation/action-space
    delegation test."""

    def __init__(self, label):
        self.label = label

    def sample(self):
        return f"sampled[{self.label}]"

    def __eq__(self, other):
        return self.label == other

    def __repr__(self):
        return self.label


class FakeEnv:
    """A trivial 2-controlled / 1-opponent ParallelEnv stand-in."""

    def __init__(self):
        self.possible_agents = ["adversary_0", "adversary_1", "agent_0"]
        self.agents = list(self.possible_agents)
        self.metadata = {}
        self.render_mode = None
        self.last_step_actions = None

    def observation_space(self, agent):
        return FakeSpace(f"obs_space[{agent}]")

    def action_space(self, agent):
        return FakeSpace(f"action_space[{agent}]")

    def reset(self, seed=None, options=None):
        obs = {a: f"obs[{a}]" for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        self.last_step_actions = actions
        obs = {a: f"next_obs[{a}]" for a in self.agents}
        rewards = {a: 1.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rewards, terminations, truncations, infos


def make_wrapper(opponent_policy=None):
    return FixedOpponentWrapper(FakeEnv(), controlled_prefix="adversary_", opponent_policy=opponent_policy)


def test_possible_agents_and_agents_filtered_to_controlled_prefix():
    wrapper = make_wrapper()
    assert wrapper.possible_agents == ["adversary_0", "adversary_1"]
    assert wrapper.agents == ["adversary_0", "adversary_1"]


def test_reset_returns_only_controlled_agents():
    wrapper = make_wrapper()
    obs, infos = wrapper.reset()
    assert set(obs) == {"adversary_0", "adversary_1"}
    assert set(infos) == {"adversary_0", "adversary_1"}


def test_step_injects_random_opponent_action_when_no_policy_given():
    wrapper = make_wrapper(opponent_policy=None)
    wrapper.reset()
    obs, _rewards, _terminations, _truncations, _infos = wrapper.step(
        {"adversary_0": "a0", "adversary_1": "a1"}
    )
    full_actions = wrapper.env.last_step_actions
    assert full_actions["adversary_0"] == "a0"
    assert full_actions["adversary_1"] == "a1"
    assert full_actions["agent_0"] == "sampled[action_space[agent_0]]"
    assert set(obs) == {"adversary_0", "adversary_1"}


def test_step_uses_opponent_policy_when_given():
    calls = []

    def opponent_policy(agent, obs):
        calls.append((agent, obs))
        return "scripted_action"

    wrapper = make_wrapper(opponent_policy=opponent_policy)
    wrapper.reset()
    wrapper.step({"adversary_0": "a0", "adversary_1": "a1"})

    full_actions = wrapper.env.last_step_actions
    assert full_actions["agent_0"] == "scripted_action"
    assert calls == [("agent_0", "obs[agent_0]")]


def test_step_output_filtered_to_controlled_agents_only():
    wrapper = make_wrapper()
    wrapper.reset()
    obs, rewards, terminations, truncations, infos = wrapper.step(
        {"adversary_0": "a0", "adversary_1": "a1"}
    )
    for d in (obs, rewards, terminations, truncations, infos):
        assert set(d) == {"adversary_0", "adversary_1"}


def test_observation_and_action_space_delegate_to_underlying_env():
    wrapper = make_wrapper()
    assert wrapper.observation_space("adversary_0") == "obs_space[adversary_0]"
    assert wrapper.action_space("adversary_0") == "action_space[adversary_0]"
