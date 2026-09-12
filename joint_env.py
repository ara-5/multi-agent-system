"""Flatten a fully-cooperative, shared-reward PettingZoo ParallelEnv with
heterogeneous agents into a single-agent gymnasium.Env, so a single PPO policy
outputs a joint action for every agent at once.

This is a centralized-training/centralized-execution shortcut. It's appropriate
here (unlike simple_tag) because the task is fully cooperative with one shared
reward, so there's no decentralization requirement to preserve. It also sidesteps
a real problem with the freeze-one-side approach used for simple_tag: if you
freeze a random speaker while training the listener (or vice versa), the message
channel carries no information about the target, so no communication protocol can
possibly emerge -- the two roles have to co-adapt simultaneously, which a single
joint policy gets "for free" by construction.
"""
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class JointPolicyEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(self, pz_env_fn):
        super().__init__()
        self.pz_env = pz_env_fn()  # the raw PettingZoo env, exposed for recording/analysis scripts
        self.agent_order = list(self.pz_env.possible_agents)
        obs_dims = [self.pz_env.observation_space(a).shape[0] for a in self.agent_order]
        self.obs_dims = obs_dims  # per-agent obs width, in agent_order -- useful for analysis scripts
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sum(obs_dims),), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [self.pz_env.action_space(a).n for a in self.agent_order]
        )
        self.render_mode = getattr(self.pz_env, "render_mode", None)

    def _flatten_obs(self, obs_dict):
        return np.concatenate(
            [np.asarray(obs_dict[a], dtype=np.float32) for a in self.agent_order]
        )

    def reset(self, *, seed=None, options=None):
        obs, _infos = self.pz_env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), {}

    def step(self, action):
        actions = {agent: int(a) for agent, a in zip(self.agent_order, action)}
        obs, rewards, terminations, truncations, _infos = self.pz_env.step(actions)
        reward = float(rewards[self.agent_order[0]])
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        return self._flatten_obs(obs), reward, terminated, truncated, {}

    def render(self):
        return self.pz_env.render()

    def close(self):
        self.pz_env.close()
