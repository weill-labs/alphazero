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

POLICY_HEAD_FLATTEN = "flatten"
POLICY_HEAD_PER_COLUMN = "per_column"
_VALID_POLICY_HEADS = (POLICY_HEAD_FLATTEN, POLICY_HEAD_PER_COLUMN)

INPUT_EMBED_LINEAR = "linear"
INPUT_EMBED_CONV3X3 = "conv3x3"
_VALID_INPUT_EMBEDS = (INPUT_EMBED_LINEAR, INPUT_EMBED_CONV3X3)


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
    # Tier 1 transformer-only knobs (defaults preserve the v1 "naive" transformer
    # so old checkpoints load unchanged). The v2 board-aware design flips all
    # three: cls token + per-column policy + conv patch embedding (AlphaViT style).
    use_value_cls_token: bool = False
    policy_head_style: str = POLICY_HEAD_FLATTEN
    input_embed_style: str = INPUT_EMBED_LINEAR

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
            if self.policy_head_style not in _VALID_POLICY_HEADS:
                msg = (
                    f"policy_head_style must be one of {_VALID_POLICY_HEADS}, "
                    f"got {self.policy_head_style!r}"
                )
                raise ValueError(msg)
            if self.input_embed_style not in _VALID_INPUT_EMBEDS:
                msg = (
                    f"input_embed_style must be one of {_VALID_INPUT_EMBEDS}, "
                    f"got {self.input_embed_style!r}"
                )
                raise ValueError(msg)
            if (
                self.policy_head_style == POLICY_HEAD_PER_COLUMN
                and obs_shape[0] * obs_shape[1] % self.action_size != 0
            ):
                # Per-column policy assumes action_size == width, with the height
                # cells in each column aggregated to one logit.
                width = obs_shape[1]
                if self.action_size != width:
                    msg = (
                        f"policy_head_style='per_column' requires action_size "
                        f"({self.action_size}) == width ({width})"
                    )
                    raise ValueError(msg)
        object.__setattr__(self, "obs_shape", obs_shape)

    def to_dict(self) -> dict[str, int | str | bool | list[int]]:
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
            "use_value_cls_token": self.use_value_cls_token,
            "policy_head_style": self.policy_head_style,
            "input_embed_style": self.input_embed_style,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AlphaZeroNetConfig:
        obs_shape = data["obs_shape"]
        if not isinstance(obs_shape, Sequence):
            msg = f"obs_shape must be a sequence, got {type(obs_shape).__name__}"
            raise ValueError(msg)
        # Legacy checkpoints (pre-Tier-1) lack the v2 keys; defaults below
        # reproduce the v1 ("naive") transformer / original resnet path.
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
            use_value_cls_token=bool(data.get("use_value_cls_token", False)),
            policy_head_style=str(data.get("policy_head_style", POLICY_HEAD_FLATTEN)),
            input_embed_style=str(data.get("input_embed_style", INPUT_EMBED_LINEAR)),
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
    """Self-attention AlphaZero network with three opt-in Tier 1 knobs.

    Default ("v1") config matches the original naive transformer: pure linear
    input projection, mean-pool value head, flatten policy head. Flipping the
    three Tier 1 knobs in ``config`` enables the AlphaViT-style design:

    - ``input_embed_style='conv3x3'``: 3x3 conv as patch embedding (local
      pattern detection before tokenization).
    - ``use_value_cls_token=True``: prepend a learnable cls token; value head
      reads the cls token only (fixes the mean-pool value bottleneck).
    - ``policy_head_style='per_column'``: aggregate the height cells of each
      column with a shared linear, giving column-equivariant action logits
      with a strong inductive bias for the C4 action structure.

    Forward contract is unchanged: ``(policy_logits, value)``.
    """

    def __init__(self, config: AlphaZeroNetConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        height, width, planes = config.obs_shape
        d = config.d_model

        # --- Input embedding (Tier 1: conv3x3 vs linear) ---
        # Same attribute name for both modes so legacy checkpoints (pre-Tier-1)
        # load without a key-rename. The dispatch happens at forward time using
        # ``config.input_embed_style``.
        if config.input_embed_style == INPUT_EMBED_CONV3X3:
            # 3x3 conv with same padding: gives each cell a receptive field of
            # 3x3 before any attention layer sees the tokens.
            self.input_proj = nnx.Conv(
                planes, d, kernel_size=(3, 3), padding="SAME", rngs=rngs
            )
        else:
            self.input_proj = nnx.Linear(planes, d, rngs=rngs)

        # --- Positional + cls token ---
        self.row_emb = nnx.Embed(height, d, rngs=rngs)
        self.col_emb = nnx.Embed(width, d, rngs=rngs)
        if config.use_value_cls_token:
            # nnx.Param wraps a (1, d_model) learnable vector that is broadcast
            # across the batch and prepended to the board-token sequence.
            self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, d)) * 0.02)
        else:
            self.cls_token = None

        # --- Transformer stack ---
        self.layers = nnx.List(
            [
                TransformerBlock(d, config.num_heads, config.mlp_dim, rngs=rngs)
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = nnx.LayerNorm(d, rngs=rngs)

        # --- Policy head (Tier 1: per_column vs flatten) ---
        # Same attribute name for both modes so legacy checkpoints load. The
        # two modes use different input/output shapes; forward dispatches on
        # ``config.policy_head_style``.
        if config.policy_head_style == POLICY_HEAD_PER_COLUMN:
            # Per-column shared head: flatten 6 cells of each column to a
            # (6*d_model)-vector, then a shared linear -> 1 logit per column.
            # ``action_size`` == ``width`` is enforced by config validation.
            self.policy_head = nnx.Linear(height * d, 1, rngs=rngs)
        else:
            self.policy_head = nnx.Linear(
                height * width * d, config.action_size, rngs=rngs
            )

        # --- Value head: always MLP -> tanh; source token differs by knob ---
        self.value_hidden = nnx.Linear(d, d, rngs=rngs)
        self.value_out = nnx.Linear(d, 1, rngs=rngs)

    def __call__(self, obs_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = obs_batch.astype(jnp.float32)
        batch, height, width, _ = x.shape

        # --- Input embedding -> tokens ---
        if self.config.input_embed_style == INPUT_EMBED_CONV3X3:
            # Conv keeps spatial layout; tokenize after.
            x = self.input_proj(x)  # (B, H, W, d_model)
            x = x.reshape((batch, height * width, -1))
        else:
            # Linear projection per cell.
            x = x.reshape((batch, height * width, -1))
            x = self.input_proj(x)

        # Position embedding for the 42 board tokens.
        row_e = self.row_emb(jnp.arange(height))  # (H, d_model)
        col_e = self.col_emb(jnp.arange(width))  # (W, d_model)
        pos = (row_e[:, None, :] + col_e[None, :, :]).reshape((height * width, -1))
        x = x + pos

        # Prepend cls token if enabled. The cls gets no positional embedding,
        # so it relies on attention to gather positional info from the board
        # tokens (matches the BERT/ViT convention).
        if self.cls_token is not None:
            cls = jnp.broadcast_to(self.cls_token[...], (batch, 1, x.shape[-1]))
            x = jnp.concatenate([cls, x], axis=1)  # (B, 1+H*W, d_model)

        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)

        # --- Split cls (if present) from board tokens ---
        if self.cls_token is not None:
            cls_out = x[:, 0, :]  # (B, d_model)
            board = x[:, 1:, :]  # (B, H*W, d_model)
        else:
            cls_out = None
            board = x

        # --- Policy head ---
        if self.config.policy_head_style == POLICY_HEAD_PER_COLUMN:
            # board: (B, H*W, d_model) -> (B, H, W, d_model)
            board_grid = board.reshape((batch, height, width, -1))
            # Reorder to (B, W, H, d_model) so each column's H cells are flat.
            per_col = jnp.transpose(board_grid, (0, 2, 1, 3))
            per_col = per_col.reshape((batch, width, height * board_grid.shape[-1]))
            policy_logits = self.policy_head(per_col).reshape((batch, width))
        else:
            policy_logits = self.policy_head(board.reshape((batch, -1)))

        # --- Value head: read cls when available, else mean-pool board ---
        value_src = cls_out if cls_out is not None else jnp.mean(board, axis=1)
        value = jax.nn.relu(self.value_hidden(value_src))
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
