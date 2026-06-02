"""Lightweight vs-random evaluation for a live strength signal during training.

The net plays greedy (argmax of the masked policy — no search, so it's cheap) vs
a uniform-random opponent over a batch of pgx games. It saturates
once the net is strong, but it's a sharp *early* signal: a run that isn't
learning sits near 50%, a run that is climbs toward ~100%. Pure pgx/JAX, so it
adds no dependency and runs batched on-device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx

from jaxzero.game_specs import DEFAULT_GAME, resolve_game
from jaxzero.net import AlphaZeroNet, apply_model


def make_evaluator(
    graphdef: nnx.GraphDef[AlphaZeroNet],
    *,
    num_games: int,
    max_steps: int,
    game: str = DEFAULT_GAME,
):
    """Return a jitted ``params, rng_key -> net_return[num_games]`` evaluator.

    Half the games assign the net to player 0 and half to player 1 (fairness).
    ``net_return`` is the net's terminal reward per game in {-1, 0, 1} (rewards
    are 0 until terminal, so summing over the no-auto-reset rollout yields it).
    """
    env = pgx.make(resolve_game(game).env_id)
    net_player = jnp.arange(num_games) % 2
    neg_inf = jnp.finfo(jnp.float32).min
    game_index = jnp.arange(num_games)

    @jax.jit
    def play(params: nnx.State, rng_key: jax.Array) -> jax.Array:
        def step(carry, step_key):
            state, net_return = carry
            logits, _ = apply_model(graphdef, params, state.observation)
            net_action = jnp.argmax(
                jnp.where(state.legal_action_mask, logits, neg_inf), axis=-1
            )
            rand_logits = jnp.where(state.legal_action_mask, 0.0, neg_inf)
            rand_action = jax.vmap(jax.random.categorical)(
                jax.random.split(step_key, num_games), rand_logits
            )
            action = jnp.where(
                state.current_player == net_player, net_action, rand_action
            )
            state = jax.vmap(env.step)(state, action)
            net_return = net_return + state.rewards[game_index, net_player]
            return (state, net_return), None

        rng_key, init_key = jax.random.split(rng_key)
        state = jax.vmap(env.init)(jax.random.split(init_key, num_games))
        (_, net_return), _ = jax.lax.scan(
            step,
            (state, jnp.zeros(num_games)),
            jax.random.split(rng_key, max_steps),
        )
        return net_return

    return play


def vs_random_metrics(net_return: jax.Array) -> dict[str, float]:
    return {
        "eval/vs_random_win_rate": float(jnp.mean(net_return == 1.0)),
        "eval/vs_random_draw_rate": float(jnp.mean(net_return == 0.0)),
        "eval/vs_random_loss_rate": float(jnp.mean(net_return == -1.0)),
    }
