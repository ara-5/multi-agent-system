"""A vocabulary-scaled variant of mpe2's simple_speaker_listener, for the
"does this hold up past 3 messages/targets" experiment the README's Next
Steps section flags as open.

mpe2's own simple_speaker_listener.py (installed at
.venv/Lib/site-packages/mpe2/simple_speaker_listener/simple_speaker_listener.py)
hardcodes `num_landmarks = 3` as a local variable inside `Scenario.make_world`
and `world.dim_c = 3` -- neither is exposed as a constructor argument, and
`dim_c` is what mpe2's SimpleEnv uses to size the discrete communication
action space (`agent.action.c[action[0]] = 1.0`, an array of length
`world.dim_c` -- see simple_env.py). There is no supported way to get more
than 3 messages/targets out of the stock environment.

This module subclasses that same Scenario (reusing its reward() unchanged --
distance-to-goal doesn't depend on the landmark count) and overrides
make_world/reset_world to parameterize the landmark count. observation() also
has to be overridden, not just reused, for a reason only found by actually
running this at num_landmarks=5 and checking the output shapes (the speaker's
observation stayed a 3-vector instead of growing to 5): the stock scenario's
speaker observation is `goal_color`, the goal landmark's *rendered RGB color*
(always `world.dim_color == 3`, a World-class default tied to pygame
rendering elsewhere -- see core.py -- and not safe to repurpose), not a
one-hot over landmark identity. That only works as a stand-in for "which of 3
landmarks" because each of the 3 hardcoded colors saturates a different one
of the 3 RGB channels; it can't represent more than 3 mutually-distinguishable
categories no matter how many landmarks exist. This scenario instead gives
the speaker a true `num_landmarks`-length one-hot of the goal's index, and
keeps landmark.color as a separate (still length-3, still just for rendering)
distinguishable RGB for however many landmarks there are. Everything else
(JointPolicyEnv, bottleneck_policy.py, mi_ppo.py, eccles_ppo.py,
analyze_communication.py, causal_intervention.py) already reads
obs_dims/action_space.nvec off the env at runtime rather than hardcoding 3, so
they all work unmodified against this for any num_landmarks -- but note that
because the speaker's observation encoding changed (one-hot instead of raw
color), a num_landmarks=3 run through *this* module is not a bit-identical
rerun of the stock env; treat it as a same-task sanity check to validate
before trusting num_landmarks=5/8 results as comparable to the stock-env
3-landmark numbers reported elsewhere in this README.
"""
from __future__ import annotations

import colorsys

import numpy as np
from gymnasium.utils import EzPickle
from mpe2._mpe_utils.core import Landmark
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
from mpe2.simple_speaker_listener.simple_speaker_listener import ExtendedAgent, ExtendedWorld, Scenario
from pettingzoo.utils.conversions import parallel_wrapper_fn


def _distinguishable_colors(n: int) -> list[np.ndarray]:
    """n evenly-spaced hues at fixed saturation/value, matching the rough
    brightness of the 3 hardcoded colors in the original scenario (each an
    RGB triple with one channel at ~0.65 and the others at ~0.15)."""
    return [np.array(colorsys.hsv_to_rgb(i / n, 0.77, 0.65)) for i in range(n)]


class ScaledScenario(Scenario):
    def __init__(self, num_landmarks: int) -> None:
        self.num_landmarks = num_landmarks

    def make_world(self) -> ExtendedWorld:
        world = ExtendedWorld()
        world.dim_c = self.num_landmarks  # discrete message vocabulary size, tied 1:1 to landmark count
        world.collaborative = True
        world.agents = [ExtendedAgent() for _ in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = "speaker_0" if i == 0 else "listener_0"
            agent.collide = False
            agent.size = 0.075
        world.agents[0].movable = False
        world.agents[1].silent = True

        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        return world

    def reset_world(self, world: ExtendedWorld, np_random: np.random.Generator) -> None:
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = world.landmarks[int(np_random.integers(len(world.landmarks)))]
        for agent in world.agents:
            agent.color = np.array([0.25, 0.25, 0.25])
        colors = _distinguishable_colors(self.num_landmarks)
        for i, landmark in enumerate(world.landmarks):
            landmark.color = colors[i]
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def observation(self, agent: ExtendedAgent, world: ExtendedWorld) -> np.ndarray:
        # speaker: a true one-hot over landmark identity (see module docstring
        # for why the stock scenario's RGB-color observation can't scale past
        # 3 landmarks the way this needs to).
        if not agent.movable:
            goal_one_hot = np.zeros(self.num_landmarks, dtype=np.float32)
            if agent._goal_b is not None:
                goal_one_hot[world.landmarks.index(agent.goal_b)] = 1.0
            return goal_one_hot

        # listener: unchanged from the stock scenario (self velocity, all
        # landmark relative positions, received communication).
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        comm = [other.state.c for other in world.agents if other is not agent and other.state.c is not None]
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
        raise RuntimeError(f"Unhandled agent role for {agent.name}.")


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_landmarks: int = 3,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        render_mode: str | None = None,
        dynamic_rescaling: bool = True,
    ) -> None:
        EzPickle.__init__(
            self, num_landmarks=num_landmarks, max_cycles=max_cycles,
            continuous_actions=continuous_actions, render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
        )
        scenario = ScaledScenario(num_landmarks)
        world = scenario.make_world()
        SimpleEnv.__init__(
            self, scenario=scenario, world=world, render_mode=render_mode,
            max_cycles=max_cycles, continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "scaled_speaker_listener"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
