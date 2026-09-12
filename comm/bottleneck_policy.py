"""A PPO policy for simple_speaker_listener that architecturally forces the
listener's movement to depend on the target ONLY via the speaker's message.

Why this exists: JointPolicyEnv flattens speaker + listener into one Gym env
with a single concatenated observation (speaker's goal one-hot followed by the
listener's own observation) and hands that whole vector to one MlpPolicy. The
default SB3 MlpPolicy runs the *entire* observation through one shared trunk
before a single linear layer produces all the action logits -- so the layer
that outputs the listener's movement logits has direct, unrestricted access to
the speaker's goal one-hot too. There is nothing in that architecture that
requires the message to carry the goal information: the network can (and, per
the low mutual-information measurements in analyze_communication.py, does)
just read the goal directly and route it to the movement logits, leaving the
message free to be noise. More training or an entropy bonus on top of that
architecture can only ever produce an incidental, weak correlation -- it can't
create a real information bottleneck that isn't there.

This policy fixes that at the architecture level instead: the message logits
are computed from a sub-network that only ever sees the speaker's slice of the
observation, and the movement logits are computed from a separate sub-network
that only ever sees the listener's slice. There is no shared layer between
them, so the only way for target information to reach the movement logits is
for the message logits (and hence the sampled message, which is the one thing
both a real decentralized speaker and listener could exchange) to encode it.
The value function is *not* bottlenecked -- a centralized critic that sees the
full joint observation is standard practice (centralized training,
decentralized execution) and doesn't weaken the actor-side guarantee above.
"""
from functools import partial

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class _SpeakerListenerExtractor(nn.Module):
    def __init__(self, speaker_dim: int, listener_dim: int, num_messages: int, num_moves: int, hidden_dim: int):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.listener_dim = listener_dim

        self.speaker_net = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.message_head = nn.Linear(hidden_dim, num_messages)

        self.listener_net = nn.Sequential(
            nn.Linear(listener_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.movement_head = nn.Linear(hidden_dim, num_moves)

        # Centralized critic: allowed to see everything, no bottleneck needed here.
        self.value_net_body = nn.Sequential(
            nn.Linear(speaker_dim + listener_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        # latent_pi is already the final concatenated [message_logits, movement_logits];
        # action_net is replaced with Identity in the policy below so nothing remixes them.
        self.latent_dim_pi = num_messages + num_moves
        self.latent_dim_vf = hidden_dim

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        speaker_slice = features[:, : self.speaker_dim]
        listener_slice = features[:, self.speaker_dim :]
        message_logits = self.message_head(self.speaker_net(speaker_slice))
        movement_logits = self.movement_head(self.listener_net(listener_slice))
        return th.cat([message_logits, movement_logits], dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net_body(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SpeakerListenerBottleneckPolicy(ActorCriticPolicy):
    """policy_kwargs must include speaker_obs_dim and listener_obs_dim
    (JointPolicyEnv.obs_dims, in agent_order -- speaker first)."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 speaker_obs_dim: int, listener_obs_dim: int, hidden_dim: int = 64, **kwargs):
        self.speaker_obs_dim = speaker_obs_dim
        self.listener_obs_dim = listener_obs_dim
        self.hidden_dim = hidden_dim
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        num_messages, num_moves = (int(n) for n in self.action_space.nvec)
        self.mlp_extractor = _SpeakerListenerExtractor(
            self.speaker_obs_dim, self.listener_obs_dim, num_messages, num_moves, self.hidden_dim
        )

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        # No trainable mixing layer: mlp_extractor's actor output is already the
        # final per-component logits, kept separate by construction.
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            for module in (self.mlp_extractor.speaker_net, self.mlp_extractor.listener_net,
                           self.mlp_extractor.value_net_body):
                module.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for module in (self.mlp_extractor.message_head, self.mlp_extractor.movement_head):
                module.apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1))
            self.features_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
