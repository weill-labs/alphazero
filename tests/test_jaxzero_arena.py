"""Tests for jaxzero/arena.py — gating match + Elo bookkeeping."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jaxzero.arena import (
    ELO_K,
    GatingResult,
    expected_elo_score,
    gating_summary,
    make_gating_match,
    update_elo,
)
from jaxzero.net import AlphaZeroNetConfig, create_model
from jaxzero.selfplay import initial_observation_shape, make_env


def _tiny_graphdef_and_params(*, seed: int = 0, game: str = "connectfour"):
    env = make_env(game)
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(game),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=1,
    )
    return nnx.split(create_model(config, seed=seed), nnx.Param)


def test_expected_elo_score_is_one_half_at_equal_ratings() -> None:
    assert expected_elo_score(1500.0, 1500.0) == pytest.approx(0.5)


def test_expected_elo_score_is_monotone_in_rating_gap() -> None:
    # 400-point lead corresponds to 10:1 odds (expected ~10/11).
    expected = expected_elo_score(1900.0, 1500.0)
    assert expected == pytest.approx(10.0 / 11.0, rel=1e-3)
    assert expected_elo_score(1500.0, 1900.0) == pytest.approx(1.0 - expected)


def test_update_elo_no_change_when_score_matches_expectation() -> None:
    # Equal ratings expect score=0.5; a 0.5 result leaves the rating unchanged.
    assert update_elo(1500.0, 1500.0, 0.5) == pytest.approx(1500.0)


def test_update_elo_full_win_against_equal_opponent_bumps_by_half_k() -> None:
    # score=1.0, expected=0.5, so delta = k * (1 - 0.5) = ELO_K / 2.
    assert update_elo(1500.0, 1500.0, 1.0) == pytest.approx(1500.0 + ELO_K / 2)


def test_update_elo_respects_custom_k() -> None:
    assert update_elo(1500.0, 1500.0, 1.0, k=32.0) == pytest.approx(1500.0 + 16.0)


def test_gating_summary_promotes_at_threshold_boundary() -> None:
    # 6 wins / (6 + 4 losses) = 0.6 decisive winrate; threshold 0.55 promotes.
    counts = jnp.array([6, 0, 4], dtype=jnp.int32)
    result = gating_summary(counts, num_games=10, threshold=0.55)
    assert isinstance(result, GatingResult)
    assert result.wins == 6
    assert result.losses == 4
    assert result.draws == 0
    assert result.winrate == pytest.approx(0.6)
    assert result.score == pytest.approx(0.6)
    assert result.promoted == 1


def test_gating_summary_does_not_promote_below_threshold() -> None:
    counts = jnp.array([5, 0, 5], dtype=jnp.int32)
    result = gating_summary(counts, num_games=10, threshold=0.55)
    assert result.winrate == pytest.approx(0.5)
    assert result.promoted == 0


def test_gating_summary_treats_draws_only_match_as_non_promotion() -> None:
    counts = jnp.array([0, 10, 0], dtype=jnp.int32)
    result = gating_summary(counts, num_games=10, threshold=0.55)
    # No decisive games -> winrate falls back to 0.0 (not undefined).
    assert result.winrate == 0.0
    assert result.score == pytest.approx(0.5)
    assert result.promoted == 0


def test_gating_summary_rejects_bad_threshold() -> None:
    counts = jnp.array([0, 0, 0], dtype=jnp.int32)
    with pytest.raises(ValueError, match="threshold"):
        gating_summary(counts, num_games=2, threshold=1.5)


def test_gating_summary_rejects_non_positive_num_games() -> None:
    counts = jnp.array([0, 0, 0], dtype=jnp.int32)
    with pytest.raises(ValueError, match="num_games"):
        gating_summary(counts, num_games=0, threshold=0.5)


def test_make_gating_match_rejects_odd_num_games() -> None:
    graphdef, _ = _tiny_graphdef_and_params()
    with pytest.raises(ValueError, match="even"):
        make_gating_match(graphdef, num_games=3, max_steps=4)


def test_make_gating_match_rejects_non_positive_args() -> None:
    graphdef, _ = _tiny_graphdef_and_params()
    with pytest.raises(ValueError, match="num_games"):
        make_gating_match(graphdef, num_games=0, max_steps=4)
    with pytest.raises(ValueError, match="max_steps"):
        make_gating_match(graphdef, num_games=2, max_steps=0)


def test_make_gating_match_self_play_is_an_even_match() -> None:
    """When candidate and best are the same net, wins + losses + draws == num_games."""
    graphdef, params = _tiny_graphdef_and_params(seed=0)
    play = make_gating_match(graphdef, num_games=4, max_steps=64)
    counts = play(params, params, jax.random.PRNGKey(0))
    assert counts.shape == (3,)
    assert counts.dtype == jnp.int32
    assert int(jnp.sum(counts)) == 4


def test_make_gating_match_returns_int_counts_summing_to_num_games() -> None:
    """Different nets still produce a valid (wins, draws, losses) partition."""
    graphdef, params_a = _tiny_graphdef_and_params(seed=0)
    _, params_b = _tiny_graphdef_and_params(seed=1)
    play = make_gating_match(graphdef, num_games=4, max_steps=64)
    counts = play(params_a, params_b, jax.random.PRNGKey(0))
    total = int(jnp.sum(counts))
    assert total == 4
    wins, draws, losses = (int(v) for v in counts.tolist())
    assert wins >= 0 and draws >= 0 and losses >= 0


def test_make_gating_match_supports_othello_shape() -> None:
    graphdef, params = _tiny_graphdef_and_params(seed=0, game="othello")
    play = make_gating_match(graphdef, num_games=2, max_steps=2, game="othello")
    counts = play(params, params, jax.random.PRNGKey(0))

    assert counts.shape == (3,)
    assert counts.dtype == jnp.int32
    assert int(jnp.sum(counts)) == 2


def test_gating_match_is_deterministic_for_fixed_seed_and_params() -> None:
    """Same (params, key) -> same counts, since action selection is greedy argmax."""
    graphdef, params_a = _tiny_graphdef_and_params(seed=0)
    _, params_b = _tiny_graphdef_and_params(seed=1)
    play = make_gating_match(graphdef, num_games=4, max_steps=64)
    counts_one = play(params_a, params_b, jax.random.PRNGKey(7))
    counts_two = play(params_a, params_b, jax.random.PRNGKey(7))
    assert jnp.array_equal(counts_one, counts_two)


def test_elo_round_trip_against_equal_opponent_with_alternating_results() -> None:
    """Win then loss against an equal opponent should be a no-op modulo K-factor."""
    start = 1500.0
    bumped = update_elo(start, start, 1.0)
    # The "opponent" is also at the original rating; a subsequent loss returns
    # toward the original rating but not exactly (since the rating moved).
    back = update_elo(bumped, start, 0.0)
    # The two updates compose to a net move smaller than K (since the second
    # update uses a *higher* expected score for the now-stronger player).
    assert math.isclose(back, start, abs_tol=ELO_K)
    assert back < bumped
