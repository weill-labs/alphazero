"""Training utilities for AlphaZero policy/value networks."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from alphazero.game import Game
from alphazero.selfplay import SelfPlayExample, TemperatureSchedule, play_game

Metrics = dict[str, float | int | str]
TimingHook = Callable[[str, float], None]


class ReplayBuffer:
    """Bounded FIFO buffer for self-play examples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._examples: deque[SelfPlayExample] = deque(maxlen=capacity)

    def add(self, examples: Sequence[SelfPlayExample]) -> None:
        self._examples.extend(examples)

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> list[SelfPlayExample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._examples:
            raise ValueError("cannot sample from an empty replay buffer")

        examples = list(self._examples)
        if batch_size >= len(examples):
            return examples

        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.choice(len(examples), size=batch_size, replace=False)
        return [examples[int(i)] for i in indices]

    def as_list(self) -> list[SelfPlayExample]:
        return list(self._examples)

    def __len__(self) -> int:
        return len(self._examples)


def make_optimizer(
    net: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Create an Adam or SGD optimizer for `net`."""

    if lr <= 0:
        raise ValueError("lr must be positive")

    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    raise ValueError("optimizer_name must be 'adam' or 'sgd'")


def loss_components(
    net: nn.Module,
    examples: Sequence[SelfPlayExample],
    l2_reg: float = 0.0,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """Return AlphaZero policy, value, L2, and total loss tensors."""

    if l2_reg < 0:
        raise ValueError("l2_reg must be non-negative")

    states, target_pi, target_value = examples_to_tensors(examples, net, device)
    policy_logits, value = net(states)
    if policy_logits.shape != target_pi.shape:
        raise ValueError(
            f"policy logits shape {tuple(policy_logits.shape)} does not match target pi "
            f"shape {tuple(target_pi.shape)}"
        )
    if value.shape != target_value.shape:
        raise ValueError(
            f"value shape {tuple(value.shape)} does not match target value shape "
            f"{tuple(target_value.shape)}"
        )

    policy_loss = -(target_pi * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    l2_loss = _l2_penalty(net, target_value.device) * l2_reg
    loss = policy_loss + value_loss + l2_loss
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "l2_loss": l2_loss,
    }


def compute_loss(
    net: nn.Module,
    examples: Sequence[SelfPlayExample],
    l2_reg: float = 0.0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compatibility helper returning total loss plus named components."""

    components = loss_components(net, examples, l2_reg=l2_reg, device=device)
    return components["loss"], components


def examples_to_tensors(
    examples: Sequence[SelfPlayExample],
    net: nn.Module,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert self-play examples into network input and target tensors."""

    if not examples:
        raise ValueError("examples must not be empty")

    target_device = torch.device(device) if device is not None else _network_device(net)
    states = np.stack([example[0] for example in examples]).astype(np.float32)
    policies = np.stack([example[1] for example in examples]).astype(np.float32)
    values = np.asarray([example[2] for example in examples], dtype=np.float32)
    return (
        torch.as_tensor(states, dtype=torch.float32, device=target_device),
        torch.as_tensor(policies, dtype=torch.float32, device=target_device),
        torch.as_tensor(values, dtype=torch.float32, device=target_device),
    )


def train_iteration(
    net: nn.Module,
    examples: Sequence[SelfPlayExample],
    optimizer: torch.optim.Optimizer | None = None,
    *,
    replay_buffer: ReplayBuffer | None = None,
    batch_size: int = 32,
    epochs: int = 1,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    momentum: float = 0.9,
    l2_reg: float = 0.0,
    device: torch.device | str | None = None,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    timing_hook: TimingHook | None = None,
) -> Metrics:
    """Train `net` on self-play examples and return aggregate metrics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    target_device = torch.device(device) if device is not None else _network_device(net)
    net.to(target_device)
    net.train()

    new_examples = list(examples)
    if replay_buffer is not None:
        replay_buffer.add(new_examples)
        training_examples = replay_buffer.as_list()
    else:
        training_examples = new_examples
    if not training_examples:
        raise ValueError("training examples must not be empty")

    opt = (
        optimizer
        if optimizer is not None
        else make_optimizer(
            net, optimizer_name=optimizer_name, lr=lr, momentum=momentum
        )
    )
    generator = rng if rng is not None else np.random.default_rng()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "l2_loss": 0.0}
    batches = 0
    seen = 0

    for _ in range(epochs):
        for batch in _batches(training_examples, batch_size, shuffle, generator):
            batch_started = time.perf_counter() if timing_hook is not None else 0.0
            opt.zero_grad(set_to_none=True)
            components = loss_components(
                net, batch, l2_reg=l2_reg, device=target_device
            )
            components["loss"].backward()
            opt.step()
            if timing_hook is not None:
                timing_hook("train_step", time.perf_counter() - batch_started)

            batch_count = len(batch)
            seen += batch_count
            batches += 1
            for key in totals:
                totals[key] += _as_float(components[key]) * batch_count

    metrics: Metrics = {
        "loss": totals["loss"] / seen,
        "policy_loss": totals["policy_loss"] / seen,
        "value_loss": totals["value_loss"] / seen,
        "l2_loss": totals["l2_loss"] / seen,
        "num_examples": len(training_examples),
        "num_new_examples": len(new_examples),
        "num_batches": batches,
    }
    if replay_buffer is not None:
        metrics["replay_buffer_size"] = len(replay_buffer)
    return metrics


def run_outer_iteration(
    net: nn.Module,
    game: Game,
    replay_buffer: ReplayBuffer,
    checkpoint_path: str | Path,
    *,
    num_selfplay_games: int = 1,
    mcts_cfg: Mapping[str, object] | None = None,
    temperature_schedule: TemperatureSchedule = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    batch_size: int = 32,
    epochs: int = 1,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    momentum: float = 0.9,
    l2_reg: float = 0.0,
    device: torch.device | str | None = None,
    timing_hook: TimingHook | None = None,
) -> Metrics:
    """Run self-play, train from replay, and checkpoint the model."""

    if num_selfplay_games <= 0:
        raise ValueError("num_selfplay_games must be positive")

    generated: list[SelfPlayExample] = []
    for _ in range(num_selfplay_games):
        play_kwargs = {}
        if timing_hook is not None:
            play_kwargs["timing_hook"] = timing_hook
        generated.extend(
            play_game(net, game, mcts_cfg, temperature_schedule, **play_kwargs)
        )

    opt = (
        optimizer
        if optimizer is not None
        else make_optimizer(
            net, optimizer_name=optimizer_name, lr=lr, momentum=momentum
        )
    )
    metrics = train_iteration(
        net,
        generated,
        optimizer=opt,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        epochs=epochs,
        l2_reg=l2_reg,
        device=device,
        timing_hook=timing_hook,
    )
    metrics["self_play_examples"] = len(generated)
    save_checkpoint(net, checkpoint_path, optimizer=opt, metrics=metrics)
    metrics["checkpoint_path"] = str(checkpoint_path)
    return metrics


train_outer_iteration = run_outer_iteration
outer_iteration = run_outer_iteration


def save_checkpoint(
    net: nn.Module,
    checkpoint_path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Mapping[str, float | int | str] | None = None,
) -> None:
    """Write a model checkpoint containing model state, optimizer state, and metrics."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"model_state": net.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    torch.save(payload, path)


def _batches(
    examples: Sequence[SelfPlayExample],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> list[list[SelfPlayExample]]:
    indices = np.arange(len(examples))
    if shuffle:
        rng.shuffle(indices)
    return [
        [examples[int(i)] for i in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def _l2_penalty(net: nn.Module, device: torch.device) -> Tensor:
    penalty = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in net.parameters():
        penalty = penalty + parameter.pow(2).sum()
    return penalty


def _network_device(net: nn.Module) -> torch.device:
    try:
        return next(net.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_float(value: Tensor) -> float:
    return float(value.detach().cpu().item())
