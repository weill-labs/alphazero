"""Two-headed AlphaZero neural network.

The network is game-agnostic: callers provide the encoded board dimensions and
the action-space width from the `Game` interface.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection (no normalization).

    The skip path keeps gradients flowing through deep trunks. BatchNorm is
    intentionally omitted so the block is safe for any batch size, including the
    size-1 final batch the training loop can produce.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    """Residual trunk with policy and value heads for AlphaZero-style search."""

    def __init__(
        self,
        num_planes: int,
        board_shape: tuple[int, int],
        action_size: int,
        *,
        channels: int = 64,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        if num_planes <= 0:
            raise ValueError("num_planes must be positive")
        if len(board_shape) != 2:
            raise ValueError("board_shape must be a (height, width) tuple")
        height, width = board_shape
        if height <= 0 or width <= 0:
            raise ValueError("board_shape dimensions must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_res_blocks < 0:
            raise ValueError("num_res_blocks must be non-negative")

        self.num_planes = num_planes
        self.board_shape = (height, width)
        self.action_size = action_size
        self.channels = channels
        self.num_res_blocks = num_res_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(num_planes, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            _ResidualBlock(channels) for _ in range(num_res_blocks)
        )

        flattened_policy = 2 * height * width
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(flattened_policy, action_size),
        )

        flattened_value = height * width
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(flattened_value, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return policy logits and a current-player value for a batch."""

        expected_shape = (self.num_planes, *self.board_shape)
        if x.ndim != 4:
            raise ValueError(
                f"expected input shape (B, {expected_shape}), got {tuple(x.shape)}"
            )
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected input shape (B, {expected_shape}), got {tuple(x.shape)}"
            )

        features = self.stem(x)
        for block in self.res_blocks:
            features = block(features)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate one encoded state and return policy probabilities plus value."""

        expected_shape = (self.num_planes, *self.board_shape)
        if state_encoding.shape != expected_shape:
            raise ValueError(
                f"expected state encoding shape {expected_shape}, got {state_encoding.shape}"
            )

        policy_probs, values = self.predict_batch(
            np.expand_dims(state_encoding, axis=0)
        )
        return policy_probs[0], float(values[0])

    def predict_batch(
        self, state_encodings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of encoded states.

        Returns policy probabilities with shape ``(B, action_size)`` and values
        with shape ``(B,)`` from each state's current-player perspective.
        """

        expected_shape = (self.num_planes, *self.board_shape)
        if (
            state_encodings.ndim != 4
            or tuple(state_encodings.shape[1:]) != expected_shape
        ):
            raise ValueError(
                f"expected state encodings shape (B, {expected_shape}), "
                f"got {state_encodings.shape}"
            )

        was_training = self.training
        self.train(False)
        try:
            with torch.no_grad():
                device = next(self.parameters()).device
                x = torch.as_tensor(state_encodings, dtype=torch.float32, device=device)
                policy_logits, value = self(x)
                policy_probs = F.softmax(policy_logits, dim=-1)
                return policy_probs.cpu().numpy(), value.cpu().numpy()
        finally:
            if was_training:
                self.train()
