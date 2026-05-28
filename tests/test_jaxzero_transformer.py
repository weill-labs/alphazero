"""Transformer-architecture smoke tests for jaxzero.

These guard the resnet/transformer A/B path: same TrainingConfig surface,
polymorphic checkpoint format, divisible-head validation, and end-to-end
train + save + load roundtrip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jaxzero.net import (
    AlphaZeroNet,
    AlphaZeroNetConfig,
    TransformerNet,
    create_model,
)
from jaxzero.selfplay import initial_observation_shape, make_env
from jaxzero.train import (
    TrainingConfig,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


def _transformer_cfg(**overrides) -> AlphaZeroNetConfig:
    env = make_env()
    base = dict(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        arch="transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
    )
    base.update(overrides)
    return AlphaZeroNetConfig(**base)


def test_create_model_dispatches_on_arch() -> None:
    transformer = create_model(_transformer_cfg(), seed=0)
    assert isinstance(transformer, TransformerNet)

    env = make_env()
    resnet = create_model(
        AlphaZeroNetConfig(
            obs_shape=initial_observation_shape(),
            action_size=env.num_actions,
            arch="resnet",
            channels=8,
            num_res_blocks=1,
        ),
        seed=0,
    )
    assert isinstance(resnet, AlphaZeroNet)


def test_transformer_forward_shapes() -> None:
    cfg = _transformer_cfg()
    model = create_model(cfg, seed=0)
    obs = jnp.zeros((3, *cfg.obs_shape), dtype=jnp.float32)
    policy, value = model(obs)
    assert policy.shape == (3, cfg.action_size)
    assert value.shape == (3,)
    # Value head is tanh-bounded.
    assert jnp.all(jnp.abs(value) <= 1.0)


def test_transformer_gradients_flow_to_attention_and_mlp() -> None:
    """All params receive non-zero gradients from a single policy-loss step.

    Without this, a typo wiring the attention output (e.g., dropping it on the
    floor) would silently train only the input projection + heads.
    """
    cfg = _transformer_cfg()
    model = create_model(cfg, seed=0)
    graphdef, params = nnx.split(model, nnx.Param)

    def loss_fn(params: nnx.State) -> jax.Array:
        merged = nnx.merge(graphdef, params)
        obs = jnp.ones((2, *cfg.obs_shape), dtype=jnp.float32)
        policy_logits, value = merged(obs)
        # Touch both heads so attention + MLP + value head all receive grad.
        return jnp.mean(policy_logits**2) + jnp.mean(value**2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree.leaves(nnx.to_pure_dict(grads))
    nonzero = [bool(jnp.any(g != 0)) for g in leaves]
    assert all(nonzero), (
        f"{sum(1 for n in nonzero if not n)} param tensors got zero grad"
    )


def test_d_model_must_be_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="d_model"):
        _transformer_cfg(d_model=33, num_heads=4)


def test_transformer_arch_rejects_unknown_value() -> None:
    env = make_env()
    with pytest.raises(ValueError, match="arch must be one of"):
        AlphaZeroNetConfig(
            obs_shape=initial_observation_shape(),
            action_size=env.num_actions,
            arch="mlp",
        )


def test_transformer_checkpoint_roundtrip(tmp_path) -> None:
    cfg = _transformer_cfg()
    model = create_model(cfg, seed=0)
    obs = jnp.zeros((2, *cfg.obs_shape), dtype=jnp.float32)
    before_policy, before_value = model(obs)

    ckpt_path = tmp_path / "transformer.msgpack"
    save_checkpoint(model, ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert isinstance(loaded, TransformerNet)
    assert loaded.config.arch == "transformer"
    assert loaded.config.d_model == cfg.d_model
    assert loaded.config.num_layers == cfg.num_layers

    after_policy, after_value = loaded(obs)
    assert jnp.allclose(before_policy, after_policy, atol=1e-6)
    assert jnp.allclose(before_value, after_value, atol=1e-6)


def test_legacy_resnet_checkpoint_loads_without_arch_field(tmp_path) -> None:
    """from_dict() must default arch='resnet' for pre-transformer checkpoints."""
    legacy_payload = {
        "obs_shape": list(initial_observation_shape()),
        "action_size": make_env().num_actions,
        "channels": 8,
        "num_res_blocks": 1,
    }
    cfg = AlphaZeroNetConfig.from_dict(legacy_payload)
    assert cfg.arch == "resnet"
    # Defaults are present for transformer-specific fields too.
    assert cfg.d_model == 128
    assert cfg.num_layers == 6


def test_run_training_with_transformer_completes_one_iter(tmp_path) -> None:
    """Tiny end-to-end: self-play + grad step + checkpoint, all with arch=transformer."""
    ckpt = tmp_path / "t.msgpack"
    cfg = TrainingConfig(
        iterations=1,
        batch_size=2,
        num_simulations=2,
        max_steps=4,
        arch="transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
        learning_rate=1e-3,
        seed=0,
        checkpoint_path=str(ckpt),
    )
    result = run_training(cfg)
    assert result.net_config.arch == "transformer"
    assert ckpt.exists()
    last = result.metrics[-1]
    # Loss is finite (gradients didn't NaN).
    assert jnp.isfinite(last["loss"])


def test_run_training_with_transformer_and_mirror_augment(tmp_path) -> None:
    """mirror_augment is data-side only; must compose with the new arch."""
    cfg = TrainingConfig(
        iterations=1,
        batch_size=2,
        num_simulations=2,
        max_steps=4,
        arch="transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
        learning_rate=1e-3,
        seed=0,
        mirror_augment=True,
    )
    result = run_training(cfg)
    assert jnp.isfinite(result.metrics[-1]["loss"])


# -----------------------------------------------------------------------------
# Tier 1 transformer changes: cls token, per-column policy, conv patch embedding.
# -----------------------------------------------------------------------------


def test_cls_token_param_only_exists_when_enabled() -> None:
    """The cls_token Param must be present when use_value_cls_token=True and
    absent otherwise, so legacy checkpoints don't carry a phantom param."""
    base = create_model(_transformer_cfg(use_value_cls_token=False), seed=0)
    with_cls = create_model(_transformer_cfg(use_value_cls_token=True), seed=0)
    assert base.cls_token is None
    assert with_cls.cls_token is not None
    # cls_token shape is (1, d_model).
    assert with_cls.cls_token[...].shape == (1, 32)


def test_cls_token_receives_gradient_from_value_loss() -> None:
    """If cls is wired correctly, a value-only loss should send gradient to it.

    Catches the bug where the cls token is created but the value head still
    reads from a mean-pool — the cls would never get a learning signal.
    """
    cfg = _transformer_cfg(use_value_cls_token=True)
    model = create_model(cfg, seed=0)
    graphdef, params = nnx.split(model, nnx.Param)

    def value_only_loss(params: nnx.State) -> jax.Array:
        merged = nnx.merge(graphdef, params)
        obs = jnp.ones((2, *cfg.obs_shape), dtype=jnp.float32)
        _, value = merged(obs)
        return jnp.mean(value**2)

    grads = jax.grad(value_only_loss)(params)
    # Walk the flat state and find the cls_token gradient by its path key.
    flat = nnx.to_pure_dict(grads)
    cls_grad = flat["cls_token"]
    assert jnp.any(cls_grad != 0), "cls_token got zero gradient from value loss"


def test_per_column_policy_head_outputs_correct_shape() -> None:
    cfg = _transformer_cfg(policy_head_style="per_column")
    model = create_model(cfg, seed=0)
    obs = jnp.zeros((5, *cfg.obs_shape), dtype=jnp.float32)
    policy, _ = model(obs)
    # Connect 4: 7 columns -> action_size 7
    assert policy.shape == (5, cfg.action_size)


def test_per_column_policy_head_param_count_is_smaller_than_flatten() -> None:
    """Per-column shares params across columns: ~width factor fewer policy params."""
    flatten = create_model(_transformer_cfg(policy_head_style="flatten"), seed=0)
    per_col = create_model(_transformer_cfg(policy_head_style="per_column"), seed=0)

    def count(model):
        _, params = nnx.split(model, nnx.Param)
        return sum(
            int(jnp.prod(jnp.array(l.shape)))
            for l in jax.tree.leaves(nnx.to_pure_dict(params))
        )

    assert count(per_col) < count(flatten), "per_column should have fewer params"


def test_per_column_requires_action_size_equals_width() -> None:
    env = make_env()
    obs_shape = initial_observation_shape()
    with pytest.raises(ValueError, match="per_column"):
        AlphaZeroNetConfig(
            obs_shape=obs_shape,
            action_size=env.num_actions + 1,  # mismatch
            arch="transformer",
            d_model=32,
            num_layers=2,
            num_heads=4,
            mlp_dim=64,
            policy_head_style="per_column",
        )


def test_conv3x3_input_embed_uses_conv_not_linear() -> None:
    """conv3x3 must instantiate the Conv path and leave the Linear path None."""
    conv = create_model(_transformer_cfg(input_embed_style="conv3x3"), seed=0)
    linear = create_model(_transformer_cfg(input_embed_style="linear"), seed=0)
    assert conv.input_proj_conv is not None
    assert conv.input_proj_linear is None
    assert linear.input_proj_conv is None
    assert linear.input_proj_linear is not None


def test_invalid_policy_head_style_rejected() -> None:
    with pytest.raises(ValueError, match="policy_head_style"):
        _transformer_cfg(policy_head_style="bogus")


def test_invalid_input_embed_style_rejected() -> None:
    with pytest.raises(ValueError, match="input_embed_style"):
        _transformer_cfg(input_embed_style="bogus")


def test_v2_all_checkpoint_roundtrip(tmp_path) -> None:
    """All three Tier 1 knobs flipped: save -> load gives back identical model."""
    cfg = _transformer_cfg(
        use_value_cls_token=True,
        policy_head_style="per_column",
        input_embed_style="conv3x3",
    )
    model = create_model(cfg, seed=0)
    obs = jnp.zeros((2, *cfg.obs_shape), dtype=jnp.float32)
    before_p, before_v = model(obs)

    ckpt = tmp_path / "v2.msgpack"
    save_checkpoint(model, ckpt)
    loaded = load_checkpoint(ckpt)
    assert loaded.config.use_value_cls_token is True
    assert loaded.config.policy_head_style == "per_column"
    assert loaded.config.input_embed_style == "conv3x3"

    after_p, after_v = loaded(obs)
    assert jnp.allclose(before_p, after_p, atol=1e-6)
    assert jnp.allclose(before_v, after_v, atol=1e-6)


def test_legacy_checkpoint_defaults_to_v1_knobs() -> None:
    """from_dict() must default the Tier 1 knobs so pre-v2 checkpoints work."""
    legacy = {
        "obs_shape": list(initial_observation_shape()),
        "action_size": make_env().num_actions,
        "channels": 8,
        "num_res_blocks": 1,
        "arch": "transformer",
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_dim": 64,
    }
    cfg = AlphaZeroNetConfig.from_dict(legacy)
    assert cfg.use_value_cls_token is False
    assert cfg.policy_head_style == "flatten"
    assert cfg.input_embed_style == "linear"


def test_v2_all_end_to_end_one_iter(tmp_path) -> None:
    """v2 transformer trains for one iter without NaNs and writes a checkpoint."""
    ckpt = tmp_path / "v2_train.msgpack"
    cfg = TrainingConfig(
        iterations=1,
        batch_size=2,
        num_simulations=2,
        max_steps=4,
        arch="transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
        use_value_cls_token=True,
        policy_head_style="per_column",
        input_embed_style="conv3x3",
        learning_rate=1e-3,
        seed=0,
        checkpoint_path=str(ckpt),
    )
    result = run_training(cfg)
    assert result.net_config.use_value_cls_token is True
    assert result.net_config.policy_head_style == "per_column"
    assert result.net_config.input_embed_style == "conv3x3"
    assert ckpt.exists()
    assert jnp.isfinite(result.metrics[-1]["loss"])


def test_v2_composes_with_mirror_augment() -> None:
    """mirror_augment is data-side and must still work with v2 (per_column is
    column-shared so the head is mirror-equivariant by construction)."""
    cfg = TrainingConfig(
        iterations=1,
        batch_size=2,
        num_simulations=2,
        max_steps=4,
        arch="transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
        use_value_cls_token=True,
        policy_head_style="per_column",
        input_embed_style="conv3x3",
        mirror_augment=True,
        learning_rate=1e-3,
        seed=0,
    )
    result = run_training(cfg)
    assert jnp.isfinite(result.metrics[-1]["loss"])
