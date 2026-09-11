"""Unit tests for JointPolicyEnv's obs-flattening/action-splitting logic, using a
minimal fake cooperative ParallelEnv instead of a real PettingZoo environment."""
from gymnasium import spaces

from joint_env import JointPolicyEnv


class FakeCoopEnv:
    """A trivial 2-agent, heterogeneous, shared-reward ParallelEnv stand-in."""

    def __init__(self):
        self.possible_agents = ["speaker_0", "listener_0"]
        self.agents = list(self.possible_agents)
        self.metadata = {}
        self.render_mode = None
        self.last_step_actions = None
        self._obs_spaces = {
            "speaker_0": spaces.Box(-1.0, 1.0, shape=(3,)),
            "listener_0": spaces.Box(-1.0, 1.0, shape=(5,)),
        }
        self._action_spaces = {
            "speaker_0": spaces.Discrete(3),
            "listener_0": spaces.Discrete(5),
        }

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs = {"speaker_0": [1.0, 2.0, 3.0], "listener_0": [4.0, 5.0, 6.0, 7.0, 8.0]}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        self.last_step_actions = actions
        obs = {"speaker_0": [9.0, 9.0, 9.0], "listener_0": [8.0, 8.0, 8.0, 8.0, 8.0]}
        rewards = {"speaker_0": -3.5, "listener_0": -3.5}
        terminations = {"speaker_0": False, "listener_0": True}
        truncations = {"speaker_0": False, "listener_0": False}
        return obs, rewards, terminations, truncations, {a: {} for a in self.agents}


def make_env():
    return JointPolicyEnv(FakeCoopEnv)


def test_observation_space_is_concatenation_of_per_agent_spaces():
    env = make_env()
    assert env.observation_space.shape == (3 + 5,)


def test_action_space_is_multidiscrete_of_per_agent_action_counts():
    env = make_env()
    assert list(env.action_space.nvec) == [3, 5]


def test_reset_flattens_obs_in_agent_order():
    env = make_env()
    obs, info = env.reset()
    assert list(obs) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert info == {}


def test_step_splits_joint_action_and_flattens_result():
    env = make_env()
    env.reset()
    obs, reward, terminated, truncated, _info = env.step([1, 3])

    assert env._env.last_step_actions == {"speaker_0": 1, "listener_0": 3}
    assert list(obs) == [9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    assert reward == -3.5
    assert terminated is True  # listener_0 terminated -> any() is True
    assert truncated is False


def test_obs_dims_records_per_agent_widths_in_agent_order():
    env = make_env()
    assert env.obs_dims == [3, 5]
