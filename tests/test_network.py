"""Tests for the AlphaZero two-headed neural network."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.network import AlphaZeroNet


def test_forward_returns_policy_logits_and_bounded_value() -> None:
    torch.manual_seed(0)
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)
    x = torch.randn(4, 2, 3, 3)

    policy_logits, value = net(x)

    assert policy_logits.shape == (4, 9)
    assert value.shape == (4,)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)

    (policy_logits.sum() + value.sum()).backward()
    assert any(param.grad is not None for param in net.parameters())


def test_predict_returns_softmax_distribution_for_single_state() -> None:
    torch.manual_seed(0)
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)
    state = np.zeros((2, 3, 3), dtype=np.float32)
    state[0, 1, 1] = 1.0

    policy_probs, value = net.predict(state)

    assert policy_probs.shape == (9,)
    assert policy_probs.dtype == np.float32
    assert np.all(policy_probs >= 0.0)
    np.testing.assert_allclose(policy_probs.sum(), 1.0, rtol=1e-6)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_predict_batch_returns_softmax_distribution_for_states() -> None:
    torch.manual_seed(0)
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)
    states = np.zeros((4, 2, 3, 3), dtype=np.float32)
    states[:, 0, 1, 1] = 1.0

    policy_probs, values = net.predict_batch(states)

    assert policy_probs.shape == (4, 9)
    assert policy_probs.dtype == np.float32
    assert values.shape == (4,)
    assert values.dtype == np.float32
    assert np.all(policy_probs >= 0.0)
    np.testing.assert_allclose(policy_probs.sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(values >= -1.0)
    assert np.all(values <= 1.0)


def test_predict_runs_in_eval_mode_without_leaving_training_disabled() -> None:
    class RecordingNet(AlphaZeroNet):
        def __init__(self) -> None:
            super().__init__(num_planes=2, board_shape=(3, 3), action_size=9)
            self.training_modes: list[bool] = []

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.training_modes.append(self.training)
            return super().forward(x)

    net = RecordingNet()
    net.train()

    net.predict(np.zeros((2, 3, 3), dtype=np.float32))

    assert net.training_modes == [False]
    assert net.training


def test_predict_batch_runs_in_eval_mode_without_leaving_training_disabled() -> None:
    class RecordingNet(AlphaZeroNet):
        def __init__(self) -> None:
            super().__init__(num_planes=2, board_shape=(3, 3), action_size=9)
            self.training_modes: list[bool] = []

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.training_modes.append(self.training)
            return super().forward(x)

    net = RecordingNet()
    net.train()

    net.predict_batch(np.zeros((2, 2, 3, 3), dtype=np.float32))

    assert net.training_modes == [False]
    assert net.training


@pytest.mark.parametrize(
    ("bad_shape", "message"),
    [
        ((2, 3), "expected input shape"),
        ((1, 1, 3, 3), "expected input shape"),
        ((1, 2, 4, 3), "expected input shape"),
    ],
)
def test_forward_rejects_wrong_input_shape(
    bad_shape: tuple[int, ...], message: str
) -> None:
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)
    x = torch.zeros(bad_shape)

    with pytest.raises(ValueError, match=message):
        net(x)


def test_predict_rejects_wrong_state_shape() -> None:
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)

    with pytest.raises(ValueError, match="expected state encoding shape"):
        net.predict(np.zeros((3, 3, 2), dtype=np.float32))


def test_predict_batch_rejects_wrong_state_shape() -> None:
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)

    with pytest.raises(ValueError, match="expected state encodings shape"):
        net.predict_batch(np.zeros((2, 3, 3, 2), dtype=np.float32))
