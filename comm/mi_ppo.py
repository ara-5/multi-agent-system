"""PPO with an auxiliary loss that directly maximizes an estimate of the
mutual information between the speaker's observation and its message, instead
of (or alongside) a generic entropy bonus over the whole joint action.

Why this exists: bottleneck_policy.py's architecture fix reliably produces a
partial protocol (~55-60% of max mutual information, every seed), but a
generic entropy bonus meant to push past that local optimum turned out to be
unreliable across seeds (99.7%, 0.0%, 55.3% in a 3-seed rerun -- see Design
notes in the README). The problem with a generic entropy bonus is that it's
untargeted: SB3's `ent_coef` scales the entropy of the *entire* joint action
distribution (message + movement together), so it pushes the listener's
movement to be noisier too, for no reason related to the actual goal (getting
the speaker to use its full message vocabulary informatively).

This module adds a loss term aimed only at the message channel, using the
standard label-free mutual-information-maximization regularizer from
discriminative clustering (RIM: Krause et al. 2010; IMSAT: Hu et al. 2017):
for a batch of inputs x_i with predicted distributions p(y|x_i),
    I(x; y) >= H(mean_i p(y|x_i)) - mean_i H(p(y|x_i))
i.e. mutual information is (a lower bound on) marginal entropy of the average
prediction minus the average conditional entropy of each individual
prediction -- maximized by making each individual message confident (low
conditional entropy) while collectively using the full message vocabulary
across the batch (high marginal entropy). Here x is the speaker's observation
and y is the message; since this task's speaker observation *is* the target's
one-hot color, this is directly an estimate of I(target; message), without
needing to separately track target labels through the rollout buffer.
"""
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


def message_mutual_info_estimate(message_logits: th.Tensor) -> th.Tensor:
    probs = F.softmax(message_logits, dim=1)
    conditional_entropy = -(probs * th.log(probs + 1e-12)).sum(dim=1).mean()
    marginal_probs = probs.mean(dim=0)
    marginal_entropy = -(marginal_probs * th.log(marginal_probs + 1e-12)).sum()
    return marginal_entropy - conditional_entropy


class MIBonusPPO(PPO):
    """Identical to PPO, except `mi_coef * -message_mutual_info_estimate(...)` is
    added to the loss every minibatch. `mi_coef=0.0` (the default) makes this
    exactly equivalent to plain PPO; pass `policy_class=SpeakerListenerBottleneckPolicy`
    (or any policy exposing a `.message_logits(features)` method) to use it."""

    def __init__(self, *args, mi_coef: float = 0.0, **kwargs):
        self.mi_coef = mi_coef
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, mi_estimates = [], []
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
                mi_estimate = message_mutual_info_estimate(message_logits)
                mi_estimates.append(mi_estimate.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    - self.mi_coef * mi_estimate
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
        self.logger.record("train/message_mutual_info_estimate", sum(mi_estimates) / len(mi_estimates))
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
