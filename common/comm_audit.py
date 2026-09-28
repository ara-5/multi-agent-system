"""Environment- and architecture-agnostic core of the two audit tools this
repo runs against simple_speaker_listener specifically
(comm/causal_intervention.py, comm/mi_diagnostic_callback.py, mi_ppo.py).

Why this file exists, separate from those: a script hardcoded to
SpeakerListenerBottleneckPolicy and mpe2.simple_speaker_listener_v4 is only
useful to someone re-running *this* task. The actual reusable ideas --
(a) a label-free lower bound on message<->target mutual information usable as
    an auxiliary loss or a passive training-time diagnostic (RIM, Krause et
    al. 2010; IMSAT, Hu et al. 2017), and
(b) checking whether a receiver's action is *causally* driven by a received
    message, by intervening on the message component of its observation and
    seeing whether the action distribution actually moves (in the spirit of
    Jaques et al. 2019's counterfactual social-influence framework and
    Eccles et al. 2019's positive-listening loss, arXiv:1912.05676 Eq. 4-8) --
don't depend on this repo's specific network architecture or this specific
MPE task. Any policy that exposes the two small hooks in CommChannelPolicy
below, and any environment that can produce a stream of (features,
real-message-sent) pairs, can reuse both. This is what makes it possible to
point causal_intervention_metrics() at a different environment (e.g. mpe2's
two-way simple_reference) or a different project's policy without copying
this logic.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np
import torch as th
from torch.nn import functional as F


@runtime_checkable
class CommChannelPolicy(Protocol):
    """The only two hooks the functions below need from a policy.
    bottleneck_policy.SpeakerListenerBottleneckPolicy and
    shared_trunk_mi_policy.SharedTrunkAuxLossPolicy are two different
    architectures both implementing this same interface."""

    def message_logits(self, features: th.Tensor) -> th.Tensor:
        """Logits over the discrete message a speaker sends, for a batch of
        extracted features (the output of policy.extract_features(obs))."""
        ...

    def movement_logits_with_message_override(self, features: th.Tensor, override: th.Tensor) -> th.Tensor:
        """Recompute the *receiver's* action logits with the received-message
        component of the observation replaced by `override` (a one-hot or
        all-zero tensor of width num_messages), holding everything else
        fixed."""
        ...


def message_mutual_info_estimate(message_logits: th.Tensor) -> th.Tensor:
    """RIM (Krause, Perona & Gomes, NeurIPS 2010) / IMSAT (Hu et al., ICML
    2017) label-free lower bound on I(input; message): marginal entropy of
    the batch-averaged message distribution minus the mean per-sample
    conditional entropy. See mi_ppo.py's docstring for the full derivation.
    The single canonical implementation -- mi_ppo.py's training loss and
    mi_diagnostic_callback.py's passive logger both call this rather than
    each keeping their own copy."""
    probs = F.softmax(message_logits, dim=1)
    conditional_entropy = -(probs * th.log(probs + 1e-12)).sum(dim=1).mean()
    marginal_probs = probs.mean(dim=0)
    marginal_entropy = -(marginal_probs * th.log(marginal_probs + 1e-12)).sum()
    return marginal_entropy - conditional_entropy


def causal_intervention_metrics(
    policy: CommChannelPolicy,
    feature_message_stream: Iterable[tuple[th.Tensor, int]],
    num_messages: int,
) -> dict:
    """The environment-agnostic core of comm/causal_intervention.py.

    `feature_message_stream` yields (features, real_message) pairs -- features
    is a single-row tensor (the output of policy.extract_features(obs) for
    one timestep's observation), real_message is the discrete message the
    speaker actually sent at that step. The caller owns all environment
    rollout mechanics (this repo's version lives in comm/causal_intervention.py's
    main()); this function only needs that stream, so the same audit works
    against a different environment or a different project's policy by
    supplying a different stream, without touching this code.

    Returns argmax_flip_rate (fraction of steps where forcing a message other
    than the real one changes the receiver's greedy action -- 0 means action
    doesn't causally depend on the message, no matter what a correlational
    metric like message<->target MI says), plus mean_l1_vs_zero and
    mean_kl_vs_zero (Eccles et al. 2019 Eq. 8 and Eq. 5's estimators, computed
    against the all-zero "message removed" counterfactual, reported as
    metrics here rather than as training losses)."""
    flips = 0
    l1s_vs_zero: list[float] = []
    kls_vs_zero: list[float] = []
    n_steps = 0

    for features, real_message in feature_message_stream:
        with th.no_grad():
            real_override = F.one_hot(th.tensor([real_message]), num_messages).float()
            real_logits = policy.movement_logits_with_message_override(features, real_override)
            real_probs = F.softmax(real_logits, dim=1)
            real_argmax = int(real_logits.argmax(dim=1).item())

            zero_override = th.zeros(1, num_messages)
            zero_logits = policy.movement_logits_with_message_override(features, zero_override)
            zero_probs = F.softmax(zero_logits, dim=1)

            l1s_vs_zero.append(th.abs(real_probs - zero_probs).sum().item())
            kls_vs_zero.append(F.kl_div(zero_probs.log(), real_probs, reduction="sum").item())

            for m in range(num_messages):
                if m == real_message:
                    continue
                alt_override = F.one_hot(th.tensor([m]), num_messages).float()
                alt_logits = policy.movement_logits_with_message_override(features, alt_override)
                if int(alt_logits.argmax(dim=1).item()) != real_argmax:
                    flips += 1
                    break
        n_steps += 1

    if n_steps == 0:
        raise ValueError("feature_message_stream yielded no steps")

    return {
        "n_steps": n_steps,
        "argmax_flip_rate": flips / n_steps,
        "mean_l1_vs_zero": float(np.mean(l1s_vs_zero)),
        "mean_kl_vs_zero": float(np.mean(kls_vs_zero)),
    }


def placebo_intervention_metrics(
    policy,
    feature_slice_stream: Iterable[tuple[th.Tensor, th.Tensor]],
    slice_start: int,
    slice_end: int,
) -> dict:
    """A control condition for causal_intervention_metrics' argmax_flip_rate:
    runs the identical intervention procedure (does forcing an alternative
    value change the receiver's greedy action?) on a slice of the receiver's
    OWN observation that has nothing to do with the message -- e.g. its own
    velocity -- instead of the message slice.

    Why this control matters: a high message-slice flip rate is consistent
    with two very different explanations -- (a) the receiver's action
    genuinely depends on the message, or (b) the receiver's policy is just
    generally brittle, and flips its greedy action under *any* sufficiently
    large perturbation to its input, message or not. Without this control,
    causal_intervention_metrics' result is itself vulnerable to a milder
    version of the same correlation-vs-causation trap Lowe et al. 2019
    (arXiv:1903.05168) describe for observational metrics like MI: an
    intervention-based metric still isn't evidence of *specific* causal
    reliance unless it's compared against an intervention on something that
    plainly shouldn't matter. A protocol with a real message-specific effect
    should show a message-slice flip rate well above this placebo baseline;
    if the two are comparable, the message-slice result doesn't mean what it
    looks like it means.

    `feature_slice_stream` yields (features, real_slice_value) pairs, where
    real_slice_value is the actual value (shape [1, slice_end-slice_start])
    of the slice being perturbed at that step. Tries two alternatives per
    step -- zeroed and negated -- matching the spirit of
    causal_intervention_metrics trying every alternative message."""
    flips = 0
    n_steps = 0

    for features, real_slice_value in feature_slice_stream:
        with th.no_grad():
            real_logits = policy.movement_logits_with_slice_override(
                features, slice_start, slice_end, real_slice_value
            )
            real_argmax = int(real_logits.argmax(dim=1).item())

            for alt_value in (th.zeros_like(real_slice_value), -real_slice_value):
                alt_logits = policy.movement_logits_with_slice_override(
                    features, slice_start, slice_end, alt_value
                )
                if int(alt_logits.argmax(dim=1).item()) != real_argmax:
                    flips += 1
                    break
        n_steps += 1

    if n_steps == 0:
        raise ValueError("feature_slice_stream yielded no steps")

    return {"n_steps": n_steps, "argmax_flip_rate": flips / n_steps}
