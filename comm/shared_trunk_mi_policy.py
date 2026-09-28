"""Ablation: what does an auxiliary channel-only loss (MIBonusPPO / eccles_ppo)
actually need to work -- the architectural bottleneck (bottleneck_policy.py's
two independent sub-networks), or is the loss alone enough even on the original
single-shared-trunk architecture that train_baseline.py showed plateaus at
13.5% mutual information with no loss at all?

This is the SAME network as SB3's default MlpPolicy (one shared trunk over the
full concatenated observation, one linear action_net producing all logits at
once) -- the only difference from train_baseline.py is exposing the two extra
methods (`message_logits`, `movement_logits_with_message_override`) that
mi_ppo.py / eccles_ppo.py need to compute their auxiliary losses, by slicing
into the *columns* of the shared action_net's output instead of routing through
separate sub-networks the way bottleneck_policy.py does.

Why this matters: the auxiliary loss only ever rewards the message *columns*
of the output for correlating with the target -- it says nothing about whether
the movement columns have to route through them. Since this architecture's
movement columns still come from a trunk with direct, unrestricted access to
the speaker's raw observation (the target), gradient descent can satisfy the
loss (produce a message that correlates with the target) via the speaker side
alone, while the listener side keeps reading the target shortcut directly --
i.e. message<->target mutual information can go up without the movement action
actually depending on the message at all. That's exactly the gap between a
correlational metric and a causal one that Lowe et al. 2019
(arXiv:1903.05168) warn about in the abstract -- analyze_communication.py's
confusion-matrix MI is correlational; causal_intervention.py's argmax-flip
rate is not. Run both on a model trained with this policy to see whether they
agree.
"""
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class SharedTrunkAuxLossPolicy(ActorCriticPolicy):
    """policy_kwargs must include speaker_obs_dim (JointPolicyEnv.obs_dims[0]) --
    only the speaker's observation width is needed, to know which of the shared
    action_net's output columns are the message logits vs. movement logits."""

    def __init__(self, observation_space, action_space, lr_schedule, speaker_obs_dim: int, **kwargs):
        self.speaker_obs_dim = speaker_obs_dim
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.num_messages = int(self.action_space.nvec[0])

    def message_logits(self, features: th.Tensor) -> th.Tensor:
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self.action_net(latent_pi)[:, : self.num_messages]

    def movement_logits_with_message_override(self, features: th.Tensor, override: th.Tensor) -> th.Tensor:
        """Unlike bottleneck_policy.py, there's no separate listener sub-network
        to recompute in isolation -- the movement columns come from the same
        trunk as everything else. So the "override" has to happen at the
        observation level, before the shared trunk runs, by overwriting the
        received-message slice (the last `override.shape[-1]` entries of the
        listener's observation, per MPE's layout -- see bottleneck_policy.py's
        version of this method for the same slicing) directly in `features` and
        rerunning the whole trunk + action_net. This changes the *entire*
        joint-action distribution as a side effect (not just movement), since
        nothing here separates the two -- itself evidence of the architectural
        gap this ablation exists to demonstrate."""
        overridden = features.clone()
        num_override = override.shape[-1]
        overridden[:, -num_override:] = override
        latent_pi = self.mlp_extractor.forward_actor(overridden)
        return self.action_net(latent_pi)[:, self.num_messages :]

    def movement_logits_with_slice_override(
        self, features: th.Tensor, start: int, end: int, values: th.Tensor
    ) -> th.Tensor:
        """General form of movement_logits_with_message_override, for a
        placebo control -- see bottleneck_policy.py's version of this method
        for the full rationale. `start`/`end` index into the LISTENER's own
        observation (i.e. offset by speaker_obs_dim within `features`, same
        convention as movement_logits_with_message_override above)."""
        overridden = features.clone()
        listener_start = self.speaker_obs_dim
        overridden[:, listener_start + start : listener_start + end] = values
        latent_pi = self.mlp_extractor.forward_actor(overridden)
        return self.action_net(latent_pi)[:, self.num_messages :]
