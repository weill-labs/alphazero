"""Phase 1 JAX pipeline smoke and determinism tests."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
from flax import nnx

from jaxzero.net import AlphaZeroNetConfig, create_model
from jaxzero.selfplay import (
    SelfPlayConfig,
    SelfPlayData,
    discounted_returns,
    flatten_selfplay_data,
    initial_observation_shape,
    make_env,
    make_selfplay,
)
from jaxzero.train import (
    TrainingConfig,
    _append_to_buffer,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


def _tiny_training_config(
    *, seed: int = 0, checkpoint_path: str | None = None
) -> TrainingConfig:
    return TrainingConfig(
        iterations=1,
        batch_size=2,
        num_simulations=1,
        max_steps=2,
        channels=4,
        num_res_blocks=1,
        learning_rate=1e-2,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )


def _tree_leaves(params):
    return jax.tree.leaves(nnx.to_pure_dict(params))


def test_residual_block_has_layernorm() -> None:
    """ResidualBlock uses pre-LN so the tower is stable at higher learning rates.

    Removing LN destabilizes lr 5e-3 (the unnormalized tower collapses to uniform
    output); guard against accidental removal.
    """
    env = make_env()
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=2,
    )
    model = create_model(config, seed=0)

    for block in model.blocks:
        assert isinstance(block.norm1, nnx.LayerNorm)
        assert isinstance(block.norm2, nnx.LayerNorm)


def test_nnx_net_forward_matches_contract() -> None:
    env = make_env()
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=1,
    )
    model = create_model(config, seed=0)
    obs = jnp.zeros((3, *config.obs_shape), dtype=jnp.bool_)

    policy_logits, value = model(obs)

    assert policy_logits.shape == (3, env.num_actions)
    assert value.shape == (3,)
    assert jnp.all(value <= 1.0)
    assert jnp.all(value >= -1.0)


def test_selfplay_smoke_produces_training_data() -> None:
    env = make_env()
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=1,
    )
    graphdef, params = nnx.split(create_model(config, seed=0), nnx.Param)
    selfplay = make_selfplay(
        SelfPlayConfig(batch_size=2, num_simulations=1, max_steps=2),
        graphdef,
    )

    data = selfplay(params, jax.random.PRNGKey(0))
    flat = flatten_selfplay_data(data)

    assert data.observation.shape == (2, 2, *config.obs_shape)
    assert flat.action_weights.shape == (4, env.num_actions)
    assert jnp.allclose(jnp.sum(flat.action_weights, axis=-1), 1.0)
    assert flat.value_target.shape == (4,)
    assert flat.value_mask.shape == (4,)


def test_discounted_returns_mask_truncated_suffix() -> None:
    rewards = jnp.array([[0.0], [1.0], [0.0], [0.0]])
    discounts = jnp.array([[-1.0], [0.0], [-1.0], [-1.0]])
    terminated = jnp.array([[False], [True], [False], [False]])

    targets, mask = discounted_returns(rewards, discounts, terminated)

    assert jnp.allclose(targets[:, 0], jnp.array([-1.0, 1.0, 0.0, 0.0]))
    assert mask[:, 0].tolist() == [True, True, False, False]


def test_training_smoke_and_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "jaxzero.msgpack"
    result = run_training(_tiny_training_config(checkpoint_path=str(checkpoint)))

    model = load_checkpoint(checkpoint)
    reloaded_leaves = _tree_leaves(nnx.state(model, nnx.Param))
    result_leaves = _tree_leaves(result.params)

    assert len(result.metrics) == 1
    assert jnp.isfinite(result.metrics[0]["loss"])
    assert len(reloaded_leaves) == len(result_leaves)
    for actual, expected in zip(reloaded_leaves, result_leaves, strict=True):
        assert jnp.allclose(actual, expected)


def test_checkpoint_helpers_store_net_config(tmp_path) -> None:
    env = make_env()
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=1,
    )
    model = create_model(config, seed=0)
    checkpoint = tmp_path / "model.msgpack"

    save_checkpoint(model, checkpoint)
    loaded = load_checkpoint(checkpoint)

    assert loaded.config == config


def test_same_seed_reproduces_params() -> None:
    first = run_training(_tiny_training_config(seed=123))
    second = run_training(_tiny_training_config(seed=123))

    for actual, expected in zip(
        _tree_leaves(first.params), _tree_leaves(second.params), strict=True
    ):
        assert jnp.array_equal(actual, expected)


def test_init_checkpoint_warm_starts_from_saved_net(tmp_path) -> None:
    checkpoint = tmp_path / "warm.msgpack"
    # Save a net trained with the tiny config (channels=4).
    run_training(_tiny_training_config(seed=1, checkpoint_path=str(checkpoint)))

    # Warm-start with a config that would otherwise build a channels=8 net; the
    # net must come from the checkpoint (channels=4), proving init_checkpoint wins.
    warm = replace(
        _tiny_training_config(seed=2), channels=8, init_checkpoint=str(checkpoint)
    )
    result = run_training(warm)

    assert result.net_config.channels == 4


def test_checkpoint_every_writes_loadable_periodic_checkpoints(tmp_path) -> None:
    final = tmp_path / "final.msgpack"
    config = replace(
        _tiny_training_config(checkpoint_path=str(final)),
        iterations=4,
        checkpoint_every=2,
    )
    saved: list[str] = []
    run_training(config, on_checkpoint=saved.append)

    # Periodic saves at iterations 2 and 4, each invoking on_checkpoint.
    assert saved == [
        str(tmp_path / "iter_0002.msgpack"),
        str(tmp_path / "iter_0004.msgpack"),
    ]
    assert (tmp_path / "iter_0002.msgpack").exists()
    load_checkpoint(str(tmp_path / "iter_0002.msgpack"))  # mid-run checkpoint loads


def test_eval_interval_logs_vs_random_metrics() -> None:
    config = replace(
        _tiny_training_config(), iterations=2, eval_interval=1, eval_games=4
    )
    logged: list[dict[str, float | int]] = []
    run_training(config, on_iteration=logged.append)

    assert len(logged) == 2
    for metrics in logged:
        assert "eval/vs_random_win_rate" in metrics
        rates = (
            metrics["eval/vs_random_win_rate"],
            metrics["eval/vs_random_draw_rate"],
            metrics["eval/vs_random_loss_rate"],
        )
        assert all(0.0 <= r <= 1.0 for r in rates)
        assert abs(sum(rates) - 1.0) < 1e-6


def test_gating_logs_keys_when_enabled() -> None:
    # threshold=0.0 forces promotion every gate (winrate>=0 always true), which
    # both exercises the promotion path and pins eval/promoted=1.
    config = replace(
        _tiny_training_config(),
        iterations=2,
        gating_interval=1,
        gating_games=2,
        gating_threshold=0.0,
    )
    logged: list[dict[str, float | int]] = []
    run_training(config, on_iteration=logged.append)

    assert len(logged) == 2
    for metrics in logged:
        for key in (
            "eval/elo",
            "eval/gating_winrate",
            "eval/promoted",
            "eval/gating_wins",
            "eval/gating_draws",
            "eval/gating_losses",
            "eval/gating_score",
        ):
            assert key in metrics, f"missing {key}"
        assert metrics["eval/promoted"] == 1
        assert 0.0 <= metrics["eval/gating_score"] <= 1.0
        assert math.isfinite(metrics["eval/elo"])


def test_gating_disabled_does_not_log_gating_keys() -> None:
    # Default config has gating_interval=None — eval/* gating keys must stay out
    # of wandb to avoid polluting non-gated baseline charts.
    logged: list[dict[str, float | int]] = []
    run_training(_tiny_training_config(), on_iteration=logged.append)
    for metrics in logged:
        for key in ("eval/elo", "eval/gating_winrate", "eval/promoted"):
            assert key not in metrics


def test_gating_config_rejects_odd_gating_games() -> None:
    import pytest

    with pytest.raises(ValueError, match="even"):
        replace(
            _tiny_training_config(),
            gating_interval=1,
            gating_games=3,
        )


def test_gating_config_rejects_threshold_out_of_range() -> None:
    import pytest

    with pytest.raises(ValueError, match="gating_threshold"):
        replace(
            _tiny_training_config(),
            gating_interval=1,
            gating_threshold=1.5,
        )


def test_value_loss_weight_default_is_one() -> None:
    assert _tiny_training_config().value_loss_weight == 1.0


def test_value_loss_weight_changes_combined_loss_but_not_value_loss_metric() -> None:
    """Increasing value_loss_weight scales the gradient-relevant `loss` but
    leaves the reported `value_loss` (unweighted) untouched — so wandb curves
    stay comparable across runs with different weights."""
    import pytest

    base_config = _tiny_training_config(seed=42)
    weighted_config = replace(base_config, value_loss_weight=4.0)

    base_logged: list[dict[str, float | int]] = []
    weighted_logged: list[dict[str, float | int]] = []
    run_training(base_config, on_iteration=base_logged.append)
    run_training(weighted_config, on_iteration=weighted_logged.append)

    # Unweighted per-head losses are deterministic given seed+config, and the
    # value_loss_weight should not change which weights produce the first
    # gradient step (it scales the gradient, which then changes future
    # iterations — but on iter 0, value_loss is computed before any update).
    assert base_logged[0]["value_loss"] == pytest.approx(
        weighted_logged[0]["value_loss"], rel=1e-5
    )
    assert base_logged[0]["policy_loss"] == pytest.approx(
        weighted_logged[0]["policy_loss"], rel=1e-5
    )
    # `loss` differs by exactly (weight-1) * value_loss at iter 0.
    base_loss = base_logged[0]["loss"]
    weighted_loss = weighted_logged[0]["loss"]
    expected_delta = 3.0 * base_logged[0]["value_loss"]  # (4 - 1) * value_loss
    assert weighted_loss == pytest.approx(base_loss + expected_delta, rel=1e-5)


def test_value_loss_weight_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="value_loss_weight"):
        replace(_tiny_training_config(), value_loss_weight=0.0)
    with pytest.raises(ValueError, match="value_loss_weight"):
        replace(_tiny_training_config(), value_loss_weight=-1.0)


def _dummy_selfplay_data(n: int, value: float) -> SelfPlayData:
    return SelfPlayData(
        observation=jnp.full((n, 1), value),
        action_weights=jnp.full((n, 1), value),
        reward=jnp.full((n,), value),
        discount=jnp.full((n,), value),
        terminated=jnp.zeros((n,), dtype=jnp.bool_),
        value_target=jnp.full((n,), value),
        value_mask=jnp.ones((n,), dtype=jnp.bool_),
    )


def test_append_to_buffer_accumulates_and_caps() -> None:
    # Buffer-free: returns only the new data.
    out = _append_to_buffer(None, _dummy_selfplay_data(3, 1.0), None)
    assert out.observation.shape[0] == 3

    # With capacity: accumulate 3 + 4 = 7, cap to the most recent 5.
    buf = _append_to_buffer(None, _dummy_selfplay_data(3, 1.0), 5)
    buf = _append_to_buffer(buf, _dummy_selfplay_data(4, 2.0), 5)
    assert buf.observation.shape[0] == 5
    assert jnp.array_equal(buf.observation[:, 0], jnp.array([1.0, 2.0, 2.0, 2.0, 2.0]))


def test_run_training_with_replay_buffer_runs() -> None:
    config = replace(_tiny_training_config(), iterations=3, replay_capacity=8)
    result = run_training(config)
    assert len(result.metrics) == 3
    assert all(jnp.isfinite(m["loss"]) for m in result.metrics)


def test_run_training_extra_evaluator_merges_metrics() -> None:
    config = replace(
        _tiny_training_config(), iterations=2, eval_interval=1, eval_games=4
    )
    logged: list[dict[str, float | int]] = []
    call_count = [0]

    def fake_extra(model) -> dict[str, float]:
        call_count[0] += 1
        return {"eval/c4_blunder_rate": 0.123, "eval/c4_policy_match": 0.877}

    run_training(config, on_iteration=logged.append, extra_evaluator=fake_extra)

    assert call_count[0] == 2  # called every eval_interval=1 iteration
    for metrics in logged:
        assert metrics["eval/c4_blunder_rate"] == 0.123
        assert metrics["eval/c4_policy_match"] == 0.877


def test_jaxzero_imports_do_not_load_torch() -> None:
    code = (
        "import sys, jaxzero, jaxzero.train; raise SystemExit('torch' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
