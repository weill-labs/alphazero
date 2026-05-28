"""Flax NNX AlphaZero network for pgx Connect Four observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class AlphaZeroNetConfig:
    """Static network shape/config stored with checkpoints."""

    obs_shape: tuple[int, int, int]
    action_size: int
    channels: int = 64
    num_res_blocks: int = 5

    def __post_init__(self) -> None:
        obs_shape = tuple(int(dim) for dim in self.obs_shape)
        if len(obs_shape) != 3:
            msg = f"obs_shape must be (height, width, planes), got {obs_shape!r}"
            raise ValueError(msg)
        if any(dim <= 0 for dim in obs_shape):
            msg = f"obs_shape dimensions must be positive, got {obs_shape!r}"
            raise ValueError(msg)
        if self.action_size <= 0:
            msg = "action_size must be positive"
            raise ValueError(msg)
        if self.channels <= 0:
            msg = "channels must be positive"
            raise ValueError(msg)
        if self.num_res_blocks < 0:
            msg = "num_res_blocks must be non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "obs_shape", obs_shape)

    def to_dict(self) -> dict[str, int | list[int]]:
        return {
            "obs_shape": list(self.obs_shape),
            "action_size": self.action_size,
            "channels": self.channels,
            "num_res_blocks": self.num_res_blocks,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AlphaZeroNetConfig:
        obs_shape = data["obs_shape"]
        if not isinstance(obs_shape, Sequence):
            msg = f"obs_shape must be a sequence, got {type(obs_shape).__name__}"
            raise ValueError(msg)
        return cls(
            obs_shape=tuple(int(dim) for dim in obs_shape),
            action_size=int(data["action_size"]),
            channels=int(data["channels"]),
            num_res_blocks=int(data["num_res_blocks"]),
        )


class ResidualBlock(nnx.Module):
    """Pre-LN residual block: LN -> conv -> ReLU -> LN -> conv -> (+residual) -> ReLU.

    LayerNorm is applied per-spatial-cell over the channel axis (NHWC, last axis),
    so there is no train/eval batch-statistic skew (unlike BatchNorm). Pre-LN
    placement is what makes higher learning rates and deeper towers stable.
    """

    def __init__(self, channels: int, *, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.LayerNorm(channels, rngs=rngs)
        self.conv1 = nnx.Conv(
            channels,
            channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(channels, rngs=rngs)
        self.conv2 = nnx.Conv(
            channels,
            channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = jax.nn.relu(self.conv1(self.norm1(x)))
        x = self.conv2(self.norm2(x))
        return jax.nn.relu(x + residual)


class AlphaZeroNet(nnx.Module):
    """Residual AlphaZero network.

    Input observations use pgx's native Connect Four layout:
    ``[batch, height, width, planes]``.
    """

    def __init__(self, config: AlphaZeroNetConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        height, width, planes = config.obs_shape
        flat_policy_features = height * width * 2
        flat_value_features = height * width

        self.stem = nnx.Conv(
            planes,
            config.channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.blocks = nnx.List(
            [
                ResidualBlock(config.channels, rngs=rngs)
                for _ in range(config.num_res_blocks)
            ]
        )

        self.policy_conv = nnx.Conv(
            config.channels,
            2,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.policy_linear = nnx.Linear(
            flat_policy_features,
            config.action_size,
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(
            config.channels,
            1,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.value_hidden = nnx.Linear(
            flat_value_features,
            config.channels,
            rngs=rngs,
        )
        self.value_out = nnx.Linear(config.channels, 1, rngs=rngs)

    def __call__(self, obs_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = obs_batch.astype(jnp.float32)
        x = jax.nn.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)

        policy = jax.nn.relu(self.policy_conv(x))
        policy = policy.reshape((policy.shape[0], -1))
        policy_logits = self.policy_linear(policy)

        value = jax.nn.relu(self.value_conv(x))
        value = value.reshape((value.shape[0], -1))
        value = jax.nn.relu(self.value_hidden(value))
        value = jnp.tanh(self.value_out(value)).reshape((-1,))
        return policy_logits, value


def create_model(config: AlphaZeroNetConfig, *, seed: int = 0) -> AlphaZeroNet:
    """Create a deterministically initialized NNX model."""

    return AlphaZeroNet(config, rngs=nnx.Rngs(seed))


def apply_model(
    graphdef: nnx.GraphDef[AlphaZeroNet],
    params: nnx.State,
    obs_batch: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure forward helper used inside JAX transforms."""

    model = nnx.merge(graphdef, params)
    return model(obs_batch)
