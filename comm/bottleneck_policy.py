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

    def movement_logits_with_message_override(self, features: th.Tensor, override: th.Tensor) -> th.Tensor:
        """Recompute movement logits with the listener's received-message slice
        (the last `override.shape[-1]` entries of its raw observation, per MPE's
        simple_speaker_listener layout -- see web_demo/index.html's
        listenerMovementForMessage, which does the identical override in JS for
        the browser demo's "force the message" widget) replaced by `override`,
        holding the rest of the listener's observation fixed. `override` can be
        a one-hot (to test a specific forced message, as the web demo does) or
        all-zero (Eccles et al. 2019's "zero inputs in place of the messages"
        counterfactual, arXiv:1912.05676 Sec 3.2). Used by causal_intervention.py
        (a quantitative, many-episode version of the web demo's single-frame
        check) and eccles_ppo.py's positive-listening loss (Eq. 8 of the same
        paper) -- both need to know whether movement is *caused* by the message,
        which message<->target mutual information alone can't establish (see
        Lowe et al. 2019, arXiv:1903.05168, on purely observational metrics)."""
        listener_slice = features[:, self.speaker_dim :].clone()
        num_override = override.shape[-1]
        listener_slice[:, -num_override:] = override
        return self.movement_head(self.listener_net(listener_slice))

    def movement_logits_with_slice_override(
        self, features: th.Tensor, start: int, end: int, values: th.Tensor
    ) -> th.Tensor:
        """General form of movement_logits_with_message_override: overrides
        features[:, speaker_dim+start : speaker_dim+end] -- an arbitrary slice
        of the LISTENER's own observation, not specifically the received-
        message slice -- with `values`, holding everything else fixed.

        Why this exists: a high argmax_flip_rate from
        movement_logits_with_message_override alone doesn't prove the
        listener is specifically message-sensitive -- it could just mean the
        policy is generally brittle to *any* out-of-distribution perturbation
        of its input, message or not (an off-distribution instability
        confound, not a real causal-communication signal). This method lets
        causal_intervention.py run the identical intervention procedure on a
        *different*, non-message slice (e.g. the listener's own velocity) as
        a placebo control: if perturbing an irrelevant slice produces a
        comparably high flip rate, the message-slice result was measuring
        general brittleness, not message-specific causal reliance."""
        listener_slice = features[:, self.speaker_dim :].clone()
        listener_slice[:, start:end] = values
        return self.movement_head(self.listener_net(listener_slice))

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

    def message_logits(self, features: th.Tensor) -> th.Tensor:
        """The speaker sub-network's raw output for a batch of extracted features
        (i.e. the output of self.extract_features(obs)) -- used by mi_ppo.py's
        mutual-information auxiliary loss, which needs direct access to just the
        message channel, not the full joint action."""
        speaker_slice = features[:, : self.speaker_obs_dim]
        return self.mlp_extractor.message_head(self.mlp_extractor.speaker_net(speaker_slice))

    def movement_logits_with_message_override(self, features: th.Tensor, override: th.Tensor) -> th.Tensor:
        """See _SpeakerListenerExtractor.movement_logits_with_message_override --
        exposed here for callers (causal_intervention.py, eccles_ppo.py) that only
        have the policy object, the same way message_logits() exposes the speaker
        sub-network above."""
        return self.mlp_extractor.movement_logits_with_message_override(features, override)

    def movement_logits_with_slice_override(
        self, features: th.Tensor, start: int, end: int, values: th.Tensor
    ) -> th.Tensor:
        """See _SpeakerListenerExtractor.movement_logits_with_slice_override --
        exposed here the same way movement_logits_with_message_override is above."""
        return self.mlp_extractor.movement_logits_with_slice_override(features, start, end, values)
