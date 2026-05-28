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
