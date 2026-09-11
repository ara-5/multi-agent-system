"""Wrap a PettingZoo ParallelEnv so only one role's agents are controllable by the
outer (SB3) policy; the opposing role acts via a fixed policy (random, or a loaded
SB3 model) chosen once per rollout rather than trained simultaneously.

Stable-Baselines3 trains a single policy per env, so genuinely independent,
role-specific policies for an asymmetric task (predator vs. prey) need either a
multi-policy trainer (e.g. RLlib) or this simpler two-stage approach: freeze one
side, train the other, then optionally swap.
"""
from pettingzoo.utils.env import ParallelEnv
from stable_baselines3 import PPO


def load_opponent_policy(model_path: str | None):
    """Returns a callable (agent_name, obs) -> action, or None for a random policy."""
    if model_path is None:
        return None
    model = PPO.load(model_path)
    return lambda agent, obs: model.predict(obs, deterministic=True)[0]


class FixedOpponentWrapper(ParallelEnv):
    """Exposes only agents whose name starts with `controlled_prefix`; every other
    agent's action is supplied by `opponent_policy` (or sampled randomly)."""

    def __init__(self, env, controlled_prefix: str, opponent_policy=None):
        self.env = env
        self.controlled_prefix = controlled_prefix
        self.opponent_policy = opponent_policy
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self._last_obs = {}

    def _is_controlled(self, agent: str) -> bool:
        return agent.startswith(self.controlled_prefix)

    @property
    def possible_agents(self):
        return [a for a in self.env.possible_agents if self._is_controlled(a)]

    @property
    def agents(self):
        return [a for a in self.env.agents if self._is_controlled(a)]

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def _opponent_action(self, agent, obs):
        if self.opponent_policy is None:
            return self.env.action_space(agent).sample()
        return self.opponent_policy(agent, obs)

    def _filter(self, d: dict) -> dict:
        return {a: v for a, v in d.items() if self._is_controlled(a)}

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return self._filter(obs), self._filter(infos)

    def step(self, actions: dict):
        full_actions = dict(actions)
        for agent in self.env.agents:
            if not self._is_controlled(agent):
                full_actions[agent] = self._opponent_action(agent, self._last_obs[agent])

        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)
        self._last_obs = obs
        return (
            self._filter(obs),
            self._filter(rewards),
            self._filter(terminations),
            self._filter(truncations),
            self._filter(infos),
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
