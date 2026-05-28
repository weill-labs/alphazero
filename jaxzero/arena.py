"""Gating match + Elo bookkeeping for jaxzero training.

Mirrors the torch-era ``alphazero/arena.py`` gating loop on the JAX side:
snapshot the live learner as a "candidate", play a fixed-size heads-up match
against the current "best" net, promote on win-rate (over decisive games)
above a threshold, and bump the best net's Elo against itself by the match
score on promotion. The match is jitted and batched via pgx, like the rest of
jaxzero — each game's per-turn action selection is greedy argmax over the
masked policy (no search), which keeps a gate cheap relative to a self-play
iteration. A higher-fidelity MCTS gating would multiply per-gate cost by
``num_simulations``; the closed alphago-{ul3,1q2,1kc} bead trail shows
value-MAE is the active ceiling at the 0.08 plateau, so finer gating signal
won't move the bottleneck — gating exists to control the *self-play* source,
not as the headline metric.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pgx
from flax import nnx

from jaxzero.net import AlphaZeroNet, apply_model

ENV_ID = "connect_four"
ELO_K = 16.0  # matches alphazero.elo_ladder.DEFAULT_ELO_K


class GatingResult(NamedTuple):
    """Outcome of a single heads-up gating match (candidate vs best).

    ``winrate`` is computed over decisive games only (consistent with torch's
    ``gating_match``) so draws don't dilute the promotion signal. ``score`` is
    the standard Elo aggregate score (wins + 0.5*draws) / n_games over all
    games, used for the Elo update on promotion.
    """

    wins: int
    draws: int
    losses: int
    winrate: float
    score: float
    promoted: int


def expected_elo_score(rating: float, opponent_rating: float) -> float:
    """Standard Elo expected-score formula in [0, 1]."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def update_elo(
    rating: float,
    opponent_rating: float,
    score: float,
    *,
    k: float = ELO_K,
) -> float:
    """Return ``rating`` updated from an aggregate match ``score`` in [0, 1]."""
    return rating + k * (score - expected_elo_score(rating, opponent_rating))


def make_gating_match(
    graphdef: nnx.GraphDef[AlphaZeroNet],
    *,
    num_games: int,
    max_steps: int,
):
    """Return a jitted heads-up match function for gating.

    The returned function ``play(candidate_params, best_params, rng_key)``
    yields a length-3 ``jnp.int32`` array of ``(wins, draws, losses)`` from
    the candidate's perspective. The lanes are split evenly between two
    seatings (candidate-as-player-0 and best-as-player-0) so the first-move
    advantage is balanced across the match.
    """

    if num_games <= 0:
        raise ValueError("num_games must be positive")
    if num_games % 2 != 0:
        raise ValueError("num_games must be even (to balance seatings)")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    env = pgx.make(ENV_ID)
    candidate_player = jnp.concatenate(
        [
            jnp.zeros(num_games // 2, dtype=jnp.int32),
            jnp.ones(num_games // 2, dtype=jnp.int32),
        ]
    )
    game_index = jnp.arange(num_games)
    neg_inf = jnp.finfo(jnp.float32).min

    @jax.jit
    def play(
        candidate_params: nnx.State,
        best_params: nnx.State,
        rng_key: jax.Array,
    ) -> jax.Array:
        # rng_key seeds the per-lane pgx env init for symmetry breaking. Action
        # selection itself is greedy argmax over masked logits — deterministic
        # given (params, state), so no RNG is needed in the inner scan.
        def step(carry, _):
            state, candidate_return = carry
            cand_logits, _ = apply_model(graphdef, candidate_params, state.observation)
            best_logits, _ = apply_model(graphdef, best_params, state.observation)
            cand_action = jnp.argmax(
                jnp.where(state.legal_action_mask, cand_logits, neg_inf), axis=-1
            )
            best_action = jnp.argmax(
                jnp.where(state.legal_action_mask, best_logits, neg_inf), axis=-1
            )
            action = jnp.where(
                state.current_player == candidate_player, cand_action, best_action
            )
            state = jax.vmap(env.step)(state, action)
            candidate_return = (
                candidate_return + state.rewards[game_index, candidate_player]
            )
            return (state, candidate_return), None

        state = jax.vmap(env.init)(jax.random.split(rng_key, num_games))
        (_, candidate_return), _ = jax.lax.scan(
            step,
            (state, jnp.zeros(num_games)),
            xs=None,
            length=max_steps,
        )
        wins = jnp.sum(candidate_return == 1.0).astype(jnp.int32)
        draws = jnp.sum(candidate_return == 0.0).astype(jnp.int32)
        losses = jnp.sum(candidate_return == -1.0).astype(jnp.int32)
        return jnp.stack([wins, draws, losses])

    return play


def gating_summary(
    counts: jax.Array, *, num_games: int, threshold: float
) -> GatingResult:
    """Materialize ``(wins, draws, losses)`` counts into a typed result.

    Caller passes the ``jnp.int32[3]`` returned by ``make_gating_match``'s
    ``play``; we host-materialize once to compute Python-side derived metrics
    (``winrate``, ``score``, ``promoted``) and avoid leaking jax tracers into
    metrics dicts.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    wins, draws, losses = (int(v) for v in counts.tolist())
    decisive = wins + losses
    winrate = wins / decisive if decisive > 0 else 0.0
    score = (wins + 0.5 * draws) / num_games
    promoted = int(winrate >= threshold)
    return GatingResult(
        wins=wins,
        draws=draws,
        losses=losses,
        winrate=winrate,
        score=score,
        promoted=promoted,
    )
