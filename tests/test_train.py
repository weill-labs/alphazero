"""Tests for AlphaZero training utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from alphazero.network import AlphaZeroNet
from alphazero.train import (
    ReplayBuffer,
    compute_loss,
    examples_to_tensors,
    loss_components,
    make_optimizer,
    train_iteration,
)


class TinyTrainableNet(nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.policy_logits = nn.Parameter(torch.zeros(action_size))
        self.value_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        policy = self.policy_logits.unsqueeze(0).expand(batch_size, -1)
        value = torch.tanh(self.value_logit).expand(batch_size)
        return policy, value


def _synthetic_examples(count: int = 4):
    state = np.zeros((2, 3, 3), dtype=np.float32)
    policy = np.zeros(9, dtype=np.float32)
    policy[2] = 1.0
    return [(state.copy(), policy.copy(), 1) for _ in range(count)]


def test_one_train_step_reduces_loss_on_tiny_synthetic_batch() -> None:
    net = TinyTrainableNet(action_size=9)
    examples = _synthetic_examples()
    optimizer = make_optimizer(net, optimizer_name="sgd", lr=0.5, momentum=0.0)

    before, _ = compute_loss(net, examples)
    metrics = train_iteration(
        net,
        examples,
        optimizer=optimizer,
        batch_size=len(examples),
        epochs=1,
        shuffle=False,
    )
    after, _ = compute_loss(net, examples)

    assert torch.isfinite(before)
    assert torch.isfinite(after)
    assert after.item() < before.item()
    assert metrics["num_examples"] == len(examples)
    assert metrics["num_batches"] == 1


def test_loss_components_are_finite() -> None:
    torch.manual_seed(0)
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)
    components = loss_components(net, _synthetic_examples(2), l2_reg=1e-4)

    assert set(components) == {"loss", "policy_loss", "value_loss", "l2_loss"}
    for value in components.values():
        assert value.shape == ()
        assert torch.isfinite(value)


def test_example_tensors_line_up_with_network_shapes() -> None:
    net = AlphaZeroNet(num_planes=2, board_shape=(3, 3), action_size=9)
    examples = _synthetic_examples(3)
    states, target_pi, target_value = examples_to_tensors(examples, net)
    policy_logits, value = net(states)

    assert states.shape == (3, 2, 3, 3)
    assert target_pi.shape == (3, 9)
    assert target_value.shape == (3,)
    assert policy_logits.shape == target_pi.shape
    assert value.shape == target_value.shape


def test_replay_buffer_keeps_most_recent_examples() -> None:
    examples = _synthetic_examples(3)
    buffer = ReplayBuffer(capacity=2)

    buffer.add(examples)

    assert len(buffer) == 2
    assert buffer.as_list() == examples[-2:]
