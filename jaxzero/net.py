"""Flax NNX AlphaZero network for pgx Connect Four observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

ARCH_RESNET = "resnet"
ARCH_TRANSFORMER = "transformer"
_VALID_ARCHS = (ARCH_RESNET, ARCH_TRANSFORMER)


@dataclass(frozen=True)
class AlphaZeroNetConfig:
    """Static network shape/config stored with checkpoints.

    ``arch`` selects the tower: 'resnet' (default, preserves existing behavior)
    uses ``channels`` + ``num_res_blocks``; 'transformer' uses ``d_model``,
    ``num_layers``, ``num_heads``, ``mlp_dim`` and ignores the resnet fields.
    """

    obs_shape: tuple[int, int, int]
    action_size: int
    channels: int = 64
    num_res_blocks: int = 5
    arch: str = ARCH_RESNET
    d_model: int = 128
    num_layers: int = 6
    num_heads: int = 4
    mlp_dim: int = 512

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
        if self.arch not in _VALID_ARCHS:
            msg = f"arch must be one of {_VALID_ARCHS}, got {self.arch!r}"
            raise ValueError(msg)
        if self.arch == ARCH_TRANSFORMER:
            for name, val in (
                ("d_model", self.d_model),
                ("num_layers", self.num_layers),
                ("num_heads", self.num_heads),
                ("mlp_dim", self.mlp_dim),
            ):
                if val <= 0:
                    raise ValueError(f"{name} must be positive for transformer")
            if self.d_model % self.num_heads != 0:
                msg = (
                    f"d_model ({self.d_model}) must be divisible by "
                    f"num_heads ({self.num_heads})"
                )
                raise ValueError(msg)
        object.__setattr__(self, "obs_shape", obs_shape)

    def to_dict(self) -> dict[str, int | str | list[int]]:
        return {
            "obs_shape": list(self.obs_shape),
            "action_size": self.action_size,
            "channels": self.channels,
            "num_res_blocks": self.num_res_blocks,
            "arch": self.arch,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AlphaZeroNetConfig:
        obs_shape = data["obs_shape"]
        if not isinstance(obs_shape, Sequence):
            msg = f"obs_shape must be a sequence, got {type(obs_shape).__name__}"
            raise ValueError(msg)
        # Legacy checkpoints (pre-transformer) lack arch/d_model/etc keys; the
        # defaults below reproduce the original resnet path.
        return cls(
            obs_shape=tuple(int(dim) for dim in obs_shape),
            action_size=int(data["action_size"]),
            channels=int(data["channels"]),
            num_res_blocks=int(data["num_res_blocks"]),
            arch=str(data.get("arch", ARCH_RESNET)),
            d_model=int(data.get("d_model", 128)),
            num_layers=int(data.get("num_layers", 6)),
            num_heads=int(data.get("num_heads", 4)),
            mlp_dim=int(data.get("mlp_dim", 512)),
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


class TransformerBlock(nnx.Module):
    """Pre-LN transformer block: LN -> MHA -> +residual -> LN -> MLP -> +residual.

    Same residual placement as the resnet block (pre-norm), so deeper towers and
    higher learning rates remain stable. The MLP uses ReLU (matching the rest of
    the codebase) — GELU is the modern default but ReLU keeps activations
    consistent with the resnet baseline so any A/B win is attributable to
    self-attention vs convolution, not to the activation choice.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp_in = nnx.Linear(d_model, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_dim, d_model, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Self-attention (inputs_k/v default to inputs_q when omitted).
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp_out(jax.nn.relu(self.mlp_in(self.norm2(x))))
        return x


class TransformerNet(nnx.Module):
    """Pure self-attention AlphaZero network.

    Each board cell is one token (no convolutions), with separate learned row
    and column positional embeddings summed to give the model 2D geometry.
    Same forward contract as :class:`AlphaZeroNet`: ``(policy_logits, value)``.
    """

    def __init__(self, config: AlphaZeroNetConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        height, width, planes = config.obs_shape
        self.input_proj = nnx.Linear(planes, config.d_model, rngs=rngs)
        # Learned 2D positional embedding factored as row + col. Cheaper than a
        # full (H*W, d_model) table and respects the natural board axes.
        self.row_emb = nnx.Embed(height, config.d_model, rngs=rngs)
        self.col_emb = nnx.Embed(width, config.d_model, rngs=rngs)
        self.layers = nnx.List(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.mlp_dim,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = nnx.LayerNorm(config.d_model, rngs=rngs)
        # Policy: flatten all token features -> one linear to action logits.
        # Mirror-augment compatibility note: this head is column-position-aware
        # (it has independent weights per cell), so mirror_augment still
        # contributes useful gradient signal even though the policy head is not
        # mirror-equivariant by construction.
        self.policy_head = nnx.Linear(
            height * width * config.d_model,
            config.action_size,
            rngs=rngs,
        )
        # Value: mean-pool over tokens -> MLP -> tanh.
        self.value_hidden = nnx.Linear(config.d_model, config.d_model, rngs=rngs)
        self.value_out = nnx.Linear(config.d_model, 1, rngs=rngs)

    def __call__(self, obs_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = obs_batch.astype(jnp.float32)
        batch, height, width, _ = x.shape
        # Cells -> tokens.
        x = x.reshape((batch, height * width, -1))
        x = self.input_proj(x)
        # 2D positional embedding: row_emb[r] + col_emb[c] for cell (r, c).
        row_e = self.row_emb(jnp.arange(height))  # (H, d_model)
        col_e = self.col_emb(jnp.arange(width))  # (W, d_model)
        pos = (row_e[:, None, :] + col_e[None, :, :]).reshape((height * width, -1))
        x = x + pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)

        policy_logits = self.policy_head(x.reshape((batch, -1)))
        pooled = jnp.mean(x, axis=1)
        value = jax.nn.relu(self.value_hidden(pooled))
        value = jnp.tanh(self.value_out(value)).reshape((-1,))
        return policy_logits, value


# Either tower satisfies the same forward contract; train.py uses this alias
# for type hints (GraphDef[Net]) so both arches plug into the same training loop.
Net = AlphaZeroNet | TransformerNet


def create_model(config: AlphaZeroNetConfig, *, seed: int = 0) -> Net:
    """Create a deterministically initialized NNX model for ``config.arch``."""

    rngs = nnx.Rngs(seed)
    if config.arch == ARCH_TRANSFORMER:
        return TransformerNet(config, rngs=rngs)
    return AlphaZeroNet(config, rngs=rngs)


def apply_model(
    graphdef: nnx.GraphDef[Net],
    params: nnx.State,
    obs_batch: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure forward helper used inside JAX transforms."""

    model = nnx.merge(graphdef, params)
    return model(obs_batch)
