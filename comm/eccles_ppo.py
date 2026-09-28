"""PPO with Eccles et al. 2019's "positive signalling" and "positive listening"
auxiliary losses (arXiv:1912.05676, NeurIPS 2019), as a second, independently-
sourced baseline against mi_ppo.py's RIM/IMSAT loss -- both are channel-only
auxiliary losses meant to fix the same problem a generic entropy bonus can't
(see mi_ppo.py's docstring), but they come from different literatures and are
not the same objective, so it's worth knowing whether one, both, or neither
actually closes the gap here.

Equations transcribed directly from the paper (Sec 3.1-3.2), not from memory:

Positive signalling (Eq. 3 + Algorithm 1) -- encourages the speaker's message
to have high mutual information with its input, but via a loss with better-
behaved gradients than directly maximizing I(m;x) (which the paper reports
leads to poor solutions: I-maximizing policies are deterministic given x but
uniformly random unconditionally, and H(pi_M(.|x)) has infinite gradient at
that optimum). Instead:
    L_ps = lambda * mean_t[(H(m_t|x_t) - H_target)^2] - H(mean_t p(m_t|x_t))
pushing the *conditional* entropy toward a target value (not to 0) while still
maximizing the *marginal* entropy across the batch. H_target defaults to
log(num_messages)/2, which the paper reports low sensitivity to.
Note: Eq. 3 in the paper places the lambda weighting on the marginal-entropy
term, but Algorithm 1 (the box the paper says to actually run) places it on
the squared-distance term instead and weights the marginal term at 1 -- an
inconsistency in the paper itself between the two presentations. This
implementation follows Algorithm 1, since that's the one given as the actual
procedure.

Positive listening (Eq. 8) -- encourages the listener's action distribution to
depend on the received message, using the L1 norm between the real
message-conditioned distribution and a counterfactual computed with the
message zeroed out:
    L_pl(x_t) = -sum_a |pi_A(a|x_t) - pi_A(a|x'_t)|
The paper's setting is a multi-timestep recurrent policy, where x'_t (the
trajectory with messages removed) requires *fitting* a separate approximator
via a cross-entropy distillation loss (their Eq. 6, "Lce") because computing
the true no-message rollout isn't otherwise available. This repo's listener is
a memoryless function of the current-step observation only (no recurrence, no
trajectory), so x'_t is just a valid alternate input to the exact same
network -- computable directly via a second forward pass through the same
policy (bottleneck_policy.py / shared_trunk_mi_policy.py's
`movement_logits_with_message_override`, called with an all-zero override,
matching the paper's "zero inputs in place of the messages"). That makes the
distillation step (Eq. 6) unnecessary here; only Eq. 8 is implemented.
"""
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


def positive_signalling_loss(message_logits: th.Tensor, target_entropy: float, lam: float) -> th.Tensor:
    probs = F.softmax(message_logits, dim=1)
    conditional_entropy = -(probs * th.log(probs + 1e-12)).sum(dim=1)  # per-sample, Eq. 3's H(m_t|x_t)
    marginal_probs = probs.mean(dim=0)
    marginal_entropy = -(marginal_probs * th.log(marginal_probs + 1e-12)).sum()
    return lam * th.mean((conditional_entropy - target_entropy) ** 2) - marginal_entropy


def positive_listening_loss(real_movement_logits: th.Tensor, counterfactual_movement_logits: th.Tensor) -> th.Tensor:
    real_probs = F.softmax(real_movement_logits, dim=1)
    counterfactual_probs = F.softmax(counterfactual_movement_logits, dim=1)
    l1 = th.abs(real_probs - counterfactual_probs).sum(dim=1)  # Eq. 8's sum_a |.|
    return -l1.mean()  # Eq. 8 is already the loss (negative L1); minimizing it maximizes the L1 gap


class EcclesPPO(PPO):
    """Identical to PPO, except `ps_coef * positive_signalling_loss(...)` and
    `pl_coef * positive_listening_loss(...)` are added to the loss every
    minibatch. Both default to 0.0, making this identical to plain PPO. Pass
    `policy_class=SpeakerListenerBottleneckPolicy` or
    `SharedTrunkAuxLossPolicy` (anything exposing `.message_logits(features)`
    and `.movement_logits_with_message_override(features, override)`)."""

    def __init__(self, *args, ps_coef: float = 0.0, pl_coef: float = 0.0,
                 target_entropy_frac: float = 0.5, ps_lambda: float = 1.0, **kwargs):
        self.ps_coef = ps_coef
        self.pl_coef = pl_coef
        self.target_entropy_frac = target_entropy_frac
        self.ps_lambda = ps_lambda
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        num_messages = int(self.policy.action_space.nvec[0])
        target_entropy = self.target_entropy_frac * np.log(num_messages)

        entropy_losses, ps_losses, pl_losses = [], [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                features = self.policy.extract_features(rollout_data.observations)
                message_logits = self.policy.message_logits(features)
                ps_loss = positive_signalling_loss(message_logits, target_entropy, self.ps_lambda)
                ps_losses.append(ps_loss.item())

                batch_size = features.shape[0]
                zero_override = th.zeros(batch_size, num_messages, device=features.device)
                real_movement_logits = self.policy.movement_logits_with_message_override(
                    features, F.one_hot(actions[:, 0].long(), num_messages).float()
                )
                counterfactual_movement_logits = self.policy.movement_logits_with_message_override(
                    features, zero_override
                )
                pl_loss = positive_listening_loss(real_movement_logits, counterfactual_movement_logits)
                pl_losses.append(pl_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.ps_coef * ps_loss
                    + self.pl_coef * pl_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", sum(entropy_losses) / len(entropy_losses))
        self.logger.record("train/positive_signalling_loss", sum(ps_losses) / len(ps_losses))
        self.logger.record("train/positive_listening_loss", sum(pl_losses) / len(pl_losses))
        self.logger.record("train/policy_gradient_loss", sum(pg_losses) / len(pg_losses))
        self.logger.record("train/value_loss", sum(value_losses) / len(value_losses))
        self.logger.record("train/approx_kl", sum(approx_kl_divs) / len(approx_kl_divs))
        self.logger.record("train/clip_fraction", sum(clip_fractions) / len(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
