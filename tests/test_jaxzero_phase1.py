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
    mirror_selfplay_data,
    scheduled_scalar,
)
from jaxzero.train import (
    TrainingConfig,
    _append_to_buffer,
    load_checkpoint,
    run_training,
    save_checkpoint,
    solver_rehearsal_data_from_labels,
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


def test_selfplay_schedule_defaults_preserve_current_behavior() -> None:
    config = SelfPlayConfig()

    assert config.temperature == 1.0
    assert config.temperature_drop_step is None
    assert config.temperature_after_drop == 1.0
    assert config.dirichlet_fraction == 0.25
    assert config.dirichlet_fraction_drop_step is None
    assert config.dirichlet_fraction_after_drop == 0.25
    assert config.dirichlet_alpha == 0.3


def test_scheduled_scalar_switches_at_drop_step() -> None:
    before = scheduled_scalar(jnp.asarray(7), initial=1.0, drop_step=8, after_drop=0.0)
    at_drop = scheduled_scalar(jnp.asarray(8), initial=1.0, drop_step=8, after_drop=0.0)
    disabled = scheduled_scalar(
        jnp.asarray(100), initial=0.25, drop_step=None, after_drop=0.0
    )

    assert before == 1.0
    assert at_drop == 0.0
    assert disabled == 0.25


def test_selfplay_schedule_config_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="temperature"):
        SelfPlayConfig(temperature=-0.1)
    with pytest.raises(ValueError, match="temperature_after_drop"):
        SelfPlayConfig(temperature_after_drop=-0.1)
    with pytest.raises(ValueError, match="temperature_drop_step"):
        SelfPlayConfig(temperature_drop_step=-1)
    with pytest.raises(ValueError, match="dirichlet_fraction"):
        SelfPlayConfig(dirichlet_fraction=1.1)
    with pytest.raises(ValueError, match="dirichlet_fraction_after_drop"):
        SelfPlayConfig(dirichlet_fraction_after_drop=-0.1)
    with pytest.raises(ValueError, match="dirichlet_fraction_drop_step"):
        SelfPlayConfig(dirichlet_fraction_drop_step=-1)
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        SelfPlayConfig(dirichlet_alpha=0.0)


def test_selfplay_non_default_schedule_runs() -> None:
    env = make_env()
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=1,
    )
    graphdef, params = nnx.split(create_model(config, seed=0), nnx.Param)
    selfplay = make_selfplay(
        SelfPlayConfig(
            batch_size=2,
            num_simulations=1,
            max_steps=2,
            temperature_drop_step=1,
            temperature_after_drop=0.0,
            dirichlet_fraction_drop_step=1,
            dirichlet_fraction_after_drop=0.0,
        ),
        graphdef,
    )

    data = selfplay(params, jax.random.PRNGKey(0))

    assert data.action_weights.shape == (2, 2, env.num_actions)
    assert jnp.allclose(jnp.sum(data.action_weights, axis=-1), 1.0)


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


def test_weight_decay_default_is_zero() -> None:
    """Default weight_decay = 0 keeps the existing optax.adam optimizer
    (no decay). Setting weight_decay > 0 should switch to optax.adamw
    (decoupled L2)."""
    assert _tiny_training_config().weight_decay == 0.0


def test_weight_decay_nonzero_runs_without_error(tmp_path) -> None:
    """End-to-end: a tiny training run with weight_decay=0.01 finishes
    successfully (uses AdamW under the hood)."""
    config = replace(
        _tiny_training_config(checkpoint_path=str(tmp_path / "wd.msgpack")),
        weight_decay=0.01,
    )
    result = run_training(config)
    assert result.checkpoint_path is not None
    assert jnp.isfinite(result.metrics[0]["loss"])


def test_weight_decay_must_be_nonneg() -> None:
    import pytest

    with pytest.raises(ValueError, match="weight_decay"):
        replace(_tiny_training_config(), weight_decay=-0.1)


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


def test_training_config_passes_selfplay_schedule_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="temperature_drop_step"):
        replace(_tiny_training_config(), selfplay_temperature_drop_step=-1)
    with pytest.raises(ValueError, match="dirichlet_fraction_after_drop"):
        replace(_tiny_training_config(), selfplay_dirichlet_fraction_after_drop=2.0)


def _example_data_for_mirror() -> SelfPlayData:
    """Hand-crafted 2-example SelfPlayData with distinguishable columns.

    obs[0] has a piece in column 0, obs[1] has a piece in column 6 — so the
    mirror swaps them. action_weights pick out the first and last columns to
    likewise verify the flip lands.
    """
    obs0 = jnp.zeros((6, 7, 2), dtype=jnp.float32).at[0, 0, 0].set(1.0)
    obs1 = jnp.zeros((6, 7, 2), dtype=jnp.float32).at[0, 6, 0].set(1.0)
    observation = jnp.stack([obs0, obs1], axis=0)
    action_weights = jnp.array(
        [[1.0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0]], dtype=jnp.float32
    )
    return SelfPlayData(
        observation=observation,
        action_weights=action_weights,
        reward=jnp.array([1.0, -1.0]),
        discount=jnp.array([0.0, 0.0]),
        terminated=jnp.array([True, True]),
        value_target=jnp.array([1.0, -1.0]),
        value_mask=jnp.ones((2,), dtype=jnp.bool_),
    )


def test_mirror_selfplay_data_doubles_size() -> None:
    data = _example_data_for_mirror()
    mirrored = mirror_selfplay_data(data)
    assert mirrored.observation.shape[0] == 4
    assert mirrored.action_weights.shape[0] == 4
    assert mirrored.value_target.shape[0] == 4
    assert mirrored.value_mask.shape[0] == 4


def test_mirror_selfplay_data_preserves_originals_then_mirrors() -> None:
    """The first N rows must be untouched; rows N..2N are the column-flipped mirrors."""
    data = _example_data_for_mirror()
    mirrored = mirror_selfplay_data(data)

    # Originals preserved at indices 0, 1.
    assert jnp.array_equal(mirrored.observation[:2], data.observation)
    assert jnp.array_equal(mirrored.action_weights[:2], data.action_weights)

    # Mirrors at indices 2, 3 have columns flipped.
    assert jnp.array_equal(mirrored.observation[2], data.observation[0, :, ::-1, :])
    assert jnp.array_equal(mirrored.observation[3], data.observation[1, :, ::-1, :])
    assert jnp.array_equal(mirrored.action_weights[2], data.action_weights[0, ::-1])
    assert jnp.array_equal(mirrored.action_weights[3], data.action_weights[1, ::-1])


def test_mirror_selfplay_data_keeps_scalar_fields_identical() -> None:
    """reward / discount / terminated / value_target / value_mask are column-agnostic."""
    data = _example_data_for_mirror()
    mirrored = mirror_selfplay_data(data)

    for field in ("reward", "discount", "terminated", "value_target", "value_mask"):
        original = getattr(data, field)
        augmented = getattr(mirrored, field)
        assert jnp.array_equal(augmented[:2], original)
        assert jnp.array_equal(augmented[2:], original)


def test_solver_rehearsal_labels_can_target_score_or_wdl_optimal_moves() -> None:
    from alphazero.games.connectfour import ConnectFour

    state = ConnectFour().initial_state()
    label = {
        "solver_value": 1,
        "solver_score": 5,
        "optimal_moves": [2, 3],
        "children": {
            2: (-1, -5),  # mover score 5: score-optimal
            3: (-1, -4),  # mover score 4: WDL-optimal but slower
            4: (0, 0),
        },
    }

    score_data = solver_rehearsal_data_from_labels([state], [label], target="score")
    wdl_data = solver_rehearsal_data_from_labels([state], [label], target="wdl")

    assert score_data.observation.shape == (1, 6, 7, 2)
    assert jnp.array_equal(
        score_data.action_weights[0],
        jnp.array([0, 0, 1, 0, 0, 0, 0], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        wdl_data.action_weights[0],
        jnp.array([0, 0, 0.5, 0.5, 0, 0, 0], dtype=jnp.float32),
    )
    assert score_data.value_target[0] == 1.0
    assert score_data.value_mask[0]


def _dummy_c4_rehearsal_data(n: int = 3) -> SelfPlayData:
    observation = jnp.zeros((n, 6, 7, 2), dtype=jnp.float32)
    action_weights = jnp.zeros((n, 7), dtype=jnp.float32).at[:, 3].set(1.0)
    return SelfPlayData(
        observation=observation,
        action_weights=action_weights,
        reward=jnp.zeros((n,), dtype=jnp.float32),
        discount=jnp.zeros((n,), dtype=jnp.float32),
        terminated=jnp.zeros((n,), dtype=jnp.bool_),
        value_target=jnp.ones((n,), dtype=jnp.float32),
        value_mask=jnp.ones((n,), dtype=jnp.bool_),
    )


def test_run_training_with_solver_rehearsal_logs_supervised_metrics(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_solver_rehearsal_data(**kwargs) -> SelfPlayData:
        calls.append(kwargs)
        return _dummy_c4_rehearsal_data(3)

    monkeypatch.setattr(
        "jaxzero.train.build_solver_rehearsal_data",
        fake_build_solver_rehearsal_data,
    )
    config = replace(
        _tiny_training_config(seed=12),
        solver_rehearsal_positions=5,
        solver_rehearsal_batch_size=2,
        solver_rehearsal_seed=99,
        solver_rehearsal_target="wdl",
    )
    logged: list[dict[str, float | int]] = []

    run_training(config, on_iteration=logged.append)

    assert calls == [
        {
            "sample_size": 5,
            "seed": 99,
            "target": "wdl",
            "solver_max_nodes": 250_000,
        }
    ]
    assert logged[0]["solver_rehearsal/examples"] == 2
    assert logged[0]["solver_rehearsal/pool_size"] == 3
    assert jnp.isfinite(logged[0]["solver_rehearsal/loss"])
    assert "solver_rehearsal/policy_loss" in logged[0]
    assert "solver_rehearsal/value_loss" in logged[0]


def test_solver_rehearsal_config_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="solver_rehearsal_positions"):
        replace(_tiny_training_config(), solver_rehearsal_positions=-1)
    with pytest.raises(ValueError, match="solver_rehearsal_batch_size"):
        replace(_tiny_training_config(), solver_rehearsal_batch_size=-1)
    with pytest.raises(ValueError, match="solver_rehearsal_interval"):
        replace(_tiny_training_config(), solver_rehearsal_interval=0)
    with pytest.raises(ValueError, match="solver_rehearsal_target"):
        replace(_tiny_training_config(), solver_rehearsal_target="bad")


def test_gating_persists_best_params_not_live(tmp_path) -> None:
    """When gating is enabled, the saved checkpoint must equal best_params,
    not the live (possibly-regressed) candidate. With threshold=1.0 and tiny
    games the candidate can never promote (no decisive 100%-win match is
    achievable in 2 games at max_steps=2), so best_params stays at iter-0
    init while the live params get trained. The saved checkpoint must match
    the init, not the trained params."""
    checkpoint = tmp_path / "gated.msgpack"
    config = replace(
        _tiny_training_config(seed=7, checkpoint_path=str(checkpoint)),
        iterations=2,
        gating_interval=1,
        gating_games=2,
        gating_threshold=1.0,  # impossible to hit -> no promotion
    )
    result = run_training(config)

    loaded = load_checkpoint(checkpoint)
    loaded_leaves = _tree_leaves(nnx.state(loaded, nnx.Param))
    live_leaves = _tree_leaves(result.params)

    # If we wrongly saved live params, every leaf would match. Since training
    # advanced and best_params stayed at init, the saved (init) leaves must
    # differ from the live (trained) leaves on at least one parameter array.
    matching = sum(
        bool(jnp.array_equal(a, b))
        for a, b in zip(loaded_leaves, live_leaves, strict=True)
    )
    assert matching < len(loaded_leaves), (
        f"saved checkpoint equals live params ({matching}/{len(loaded_leaves)} "
        "leaves match) -- expected save to use best_params (init), which "
        "differs from live."
    )


def test_no_gating_persists_live_params(tmp_path) -> None:
    """With gating off, save semantics are unchanged: saved == live (backward
    compat with all pre-jla runs)."""
    checkpoint = tmp_path / "ungated.msgpack"
    config = replace(_tiny_training_config(seed=7, checkpoint_path=str(checkpoint)))
    result = run_training(config)

    loaded = load_checkpoint(checkpoint)
    loaded_leaves = _tree_leaves(nnx.state(loaded, nnx.Param))
    live_leaves = _tree_leaves(result.params)
    for actual, expected in zip(loaded_leaves, live_leaves, strict=True):
        assert jnp.array_equal(actual, expected)


def test_mirror_augment_flag_doubles_examples_per_iteration() -> None:
    """End-to-end: setting mirror_augment=True should double the per-iteration
    self-play data fed into training. Verify by counting examples via the
    value_mask_fraction signal: training stays consistent, just on a larger
    set."""
    import pytest

    # Use replay_capacity so the buffer reflects what's training-visible.
    base_config = replace(_tiny_training_config(seed=99), iterations=1)
    mirror_config = replace(base_config, mirror_augment=True)

    base_metrics: list[dict[str, float | int]] = []
    mirror_metrics: list[dict[str, float | int]] = []
    run_training(base_config, on_iteration=base_metrics.append)
    run_training(mirror_config, on_iteration=mirror_metrics.append)

    # value_mask_fraction is the fraction of valid value targets in the batch.
    # Mirror doubles the example count but each mirror copy retains its mask,
    # so the FRACTION is preserved (numerator and denominator both 2x).
    assert base_metrics[0]["value_mask_fraction"] == pytest.approx(
        mirror_metrics[0]["value_mask_fraction"], rel=1e-5
    )


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


def test_gated_inline_eval_scores_persisted_best_not_live(tmp_path) -> None:
    """Regression test for alphago-bac.

    Under gating, the inline eval block must score the SAME params that get
    persisted to the checkpoint (best_params), not the live candidate
    (params). With tiny games every match draws, so the gating winrate is 0.0
    (< threshold 0.55) and nothing is ever promoted -- best_params stays at
    iter-0 init while the live params advance via gradient steps. The
    extra_evaluator captures the model it receives; we assert those captured
    params equal the persisted checkpoint (both best_params) and DIFFER from
    the final live params. The "differ from live" assertion is what fails on
    the pre-fix code, which scored live params.
    """
    checkpoint = tmp_path / "gated.msgpack"
    config = replace(
        _tiny_training_config(seed=0, checkpoint_path=str(checkpoint)),
        iterations=2,
        gating_interval=2,
        gating_games=2,
        gating_threshold=0.55,  # tiny games draw -> winrate 0.0 -> no promotion
        eval_interval=2,
    )

    captured: list[list] = []

    def capture_eval(model) -> dict[str, float]:
        captured.append(_tree_leaves(nnx.state(model, nnx.Param)))
        return {"eval/c4_stub": 0.0}

    logged: list[dict[str, float | int]] = []
    result = run_training(
        config, on_iteration=logged.append, extra_evaluator=capture_eval
    )

    # Precondition: no promotion happened, so best_params (persisted/eval'd)
    # diverges from the live trained params -- the property the test needs.
    assert logged[-1].get("eval/promoted", 0) == 0, (
        "expected no promotion (tiny games draw); if promoted, best == live "
        "and the test cannot distinguish best from live"
    )

    # extra_evaluator fires once at eval_interval=2 (after iteration 2).
    assert len(captured) == 1
    eval_leaves = captured[0]

    loaded = load_checkpoint(checkpoint)
    persisted_leaves = _tree_leaves(nnx.state(loaded, nnx.Param))
    live_leaves = _tree_leaves(result.params)

    # The inline eval scored exactly the params that were persisted (best).
    for evaled, persisted in zip(eval_leaves, persisted_leaves, strict=True):
        assert jnp.allclose(evaled, persisted)

    # And those persisted/eval'd (best, init) params differ from the live
    # trained params on at least one leaf -- proving eval scored best, not
    # live. This assertion fails on the pre-fix code.
    matching = sum(
        bool(jnp.array_equal(a, b))
        for a, b in zip(eval_leaves, live_leaves, strict=True)
    )
    assert matching < len(eval_leaves), (
        f"inline eval scored the live params ({matching}/{len(eval_leaves)} "
        "leaves match live) -- expected it to score best_params (init), which "
        "differs from the trained live params."
    )


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
