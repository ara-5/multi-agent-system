"""Unit tests for common/comm_audit.py's environment-agnostic metrics, using
fake policies instead of a real trained model -- the same style as
test_joint_env.py's FakeCoopEnv."""
import torch as th

from common.comm_audit import (
    causal_intervention_metrics,
    causal_intervention_metrics_by_target,
    message_mutual_info_estimate,
)


def test_message_mutual_info_estimate_is_near_zero_for_a_degenerate_speaker():
    # Always emits message 0 regardless of input -- no dependence on input at all.
    logits = th.tensor([[10.0, -10.0, -10.0]] * 8)
    estimate = message_mutual_info_estimate(logits).item()
    assert abs(estimate) < 1e-3


def test_message_mutual_info_estimate_is_near_max_for_a_clean_bijection():
    # 3 inputs, each deterministically mapped to a different one of 3 messages,
    # uniformly represented in the batch -- the maximum-MI case.
    logits = th.tensor([
        [10.0, -10.0, -10.0],
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, 10.0],
        [-10.0, -10.0, 10.0],
    ])
    estimate = message_mutual_info_estimate(logits).item()
    import math
    assert estimate > math.log(3) - 0.05  # ln(3) nats = max MI for a 3-symbol channel


class _FakePolicyNoListening:
    """The receiver's action never depends on the message override at all --
    argmax_flip_rate should be exactly 0, no matter how informative the
    message is (this is exactly the shared-trunk-ablation failure mode
    shared_trunk_mi_policy.py's docstring predicts: MI can be high while the
    receiver still doesn't causally depend on the message)."""

    def movement_logits_with_message_override(self, features: th.Tensor, override: th.Tensor) -> th.Tensor:
        return th.tensor([[1.0, 0.0, -1.0]])  # constant, ignores override entirely


class _FakePolicyRealListening:
    """The receiver's argmax action is literally the argmax of the override --
    a maximally message-dependent receiver."""

    def movement_logits_with_message_override(self, features: th.Tensor, override: th.Tensor) -> th.Tensor:
        return override * 10.0


def test_causal_intervention_metrics_flip_rate_zero_when_receiver_ignores_message():
    policy = _FakePolicyNoListening()
    stream = [(th.zeros(1, 4), message) for message in (0, 1, 2, 0, 1, 2)]
    metrics = causal_intervention_metrics(policy, stream, num_messages=3)
    assert metrics["argmax_flip_rate"] == 0.0
    assert metrics["mean_l1_vs_zero"] == 0.0


def test_causal_intervention_metrics_flip_rate_high_when_receiver_tracks_message():
    policy = _FakePolicyRealListening()
    stream = [(th.zeros(1, 4), message) for message in (0, 1, 2, 0, 1, 2)]
    metrics = causal_intervention_metrics(policy, stream, num_messages=3)
    assert metrics["argmax_flip_rate"] == 1.0
    assert metrics["mean_l1_vs_zero"] > 0.0


def test_causal_intervention_metrics_reports_n_steps():
    policy = _FakePolicyRealListening()
    stream = [(th.zeros(1, 4), 0) for _ in range(5)]
    metrics = causal_intervention_metrics(policy, stream, num_messages=3)
    assert metrics["n_steps"] == 5


def test_mean_l1_vs_alternatives_is_zero_when_receiver_ignores_message():
    policy = _FakePolicyNoListening()
    stream = [(th.zeros(1, 4), message) for message in (0, 1, 2, 0, 1, 2)]
    metrics = causal_intervention_metrics(policy, stream, num_messages=3)
    assert metrics["mean_l1_vs_alternatives"] == 0.0


def test_mean_l1_vs_alternatives_is_positive_when_receiver_tracks_message():
    policy = _FakePolicyRealListening()
    stream = [(th.zeros(1, 4), message) for message in (0, 1, 2, 0, 1, 2)]
    metrics = causal_intervention_metrics(policy, stream, num_messages=3)
    assert metrics["mean_l1_vs_alternatives"] > 0.0


def test_causal_intervention_metrics_by_target_buckets_correctly():
    policy = _FakePolicyRealListening()
    # 2 steps at target 0 (message tracks target, so always flips against the other 2
    # alternatives), 1 step at target 1 with a message that equals the target too.
    stream = [
        (th.zeros(1, 4), 0, 0),
        (th.zeros(1, 4), 0, 0),
        (th.zeros(1, 4), 1, 1),
    ]
    by_target = causal_intervention_metrics_by_target(policy, stream, num_messages=3)
    assert set(by_target.keys()) == {0, 1}
    assert by_target[0]["n_steps"] == 2
    assert by_target[1]["n_steps"] == 1
    assert by_target[0]["argmax_flip_rate"] == 1.0
    assert by_target[1]["argmax_flip_rate"] == 1.0
