"""Batched pgx + mctx Gumbel self-play for Connect Four."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import nnx
from pgx.experimental import auto_reset

from jaxzero.net import AlphaZeroNet, apply_model

ENV_ID = "connect_four"
# Root exploration noise (AlphaZero defaults). PUCT + Dirichlet makes the
# visit-count policy target non-uniform from the first game, so the policy
# learns immediately instead of waiting for the value head to bootstrap. Gumbel
# MuZero's completed-Q target collapses to uniform with an untrained value head,
# which left C4 training pinned at uniform; this matches the original PyTorch
# self-play (PUCT + Dirichlet 0.25/0.3).
_DIRICHLET_FRACTION = 0.25
_DIRICHLET_ALPHA = 0.3


@dataclass(frozen=True)
class SelfPlayConfig:
    """Static self-play settings captured by the jitted loop closure."""

    batch_size: int = 32
    num_simulations: int = 32
    max_steps: int = 64

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if self.num_simulations <= 0:
            msg = "num_simulations must be positive"
            raise ValueError(msg)
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)


class TransitionData(NamedTuple):
    observation: jax.Array
    action_weights: jax.Array
    reward: jax.Array
    discount: jax.Array
    terminated: jax.Array


class SelfPlayData(NamedTuple):
    observation: jax.Array
    action_weights: jax.Array
    reward: jax.Array
    discount: jax.Array
    terminated: jax.Array
    value_target: jax.Array
    value_mask: jax.Array


def make_env() -> pgx.Env:
    return pgx.make(ENV_ID)


def initial_observation_shape() -> tuple[int, int, int]:
    env = make_env()
    state = env.init(jax.random.PRNGKey(0))
    return tuple(int(dim) for dim in state.observation.shape)


def _mask_invalid_logits(logits: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def _clear_auto_reset_flags(state):
    return state.replace(
        terminated=jnp.zeros_like(state.terminated),
        truncated=jnp.zeros_like(state.truncated),
        rewards=jnp.zeros_like(state.rewards),
    )


def discounted_returns(
    rewards: jax.Array,
    discounts: jax.Array,
    terminated: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute return targets and masks for complete episodes only.

    Inputs are shaped ``[time, batch]``. Positions in the final incomplete
    episode of each batch lane are marked invalid for value loss.
    """

    batch_size = rewards.shape[1]

    def step(carry, transition):
        next_return, next_valid = carry
        reward, discount, done = transition
        target = reward + discount * next_return
        valid = jnp.logical_or(done, jnp.logical_and(discount != 0.0, next_valid))
        return (target, valid), (target, valid)

    init = (
        jnp.zeros((batch_size,), dtype=rewards.dtype),
        jnp.zeros((batch_size,), dtype=jnp.bool_),
    )
    _, (targets_rev, valid_rev) = jax.lax.scan(
        step,
        init,
        (rewards[::-1], discounts[::-1], terminated[::-1]),
    )
    return targets_rev[::-1], valid_rev[::-1]


def make_selfplay(
    config: SelfPlayConfig,
    graphdef: nnx.GraphDef[AlphaZeroNet],
):
    """Return a jitted self-play function ``params, rng_key -> SelfPlayData``."""

    env = make_env()
    batch_size = config.batch_size
    num_simulations = config.num_simulations
    max_steps = config.max_steps

    def recurrent_fn(params, rng_key, action, state):
        del rng_key
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = apply_model(graphdef, params, state.observation)
        logits = _mask_invalid_logits(logits, state.legal_action_mask)
        done = state.terminated | state.truncated
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(done, 0.0, value)
        discount = jnp.where(done, 0.0, -jnp.ones_like(value))
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, state

    def selfplay(params: nnx.State, rng_key: jax.Array) -> SelfPlayData:
        def step_fn(state, key):
            state = _clear_auto_reset_flags(state)
            key1, key2 = jax.random.split(key)
            logits, value = apply_model(graphdef, params, state.observation)
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=state,
            )
            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
                dirichlet_fraction=_DIRICHLET_FRACTION,
                dirichlet_alpha=_DIRICHLET_ALPHA,
            )
            current_player = state.current_player
            keys = jax.random.split(key2, batch_size)
            next_state = jax.vmap(auto_reset(env.step, env.init))(
                state,
                policy_output.action,
                keys,
            )
            done = next_state.terminated | next_state.truncated
            reward = next_state.rewards[
                jnp.arange(next_state.rewards.shape[0]), current_player
            ]
            discount = jnp.where(done, 0.0, -jnp.ones_like(reward))
            transition = TransitionData(
                observation=state.observation,
                action_weights=policy_output.action_weights,
                reward=reward,
                discount=discount,
                terminated=next_state.terminated,
            )
            return next_state, transition

        rng_key, init_key = jax.random.split(rng_key)
        state = jax.vmap(env.init)(jax.random.split(init_key, batch_size))
        _, transitions = jax.lax.scan(
            step_fn,
            state,
            jax.random.split(rng_key, max_steps),
        )
        value_target, value_mask = discounted_returns(
            transitions.reward,
            transitions.discount,
            transitions.terminated,
        )
        return SelfPlayData(
            observation=transitions.observation,
            action_weights=transitions.action_weights,
            reward=transitions.reward,
            discount=transitions.discount,
            terminated=transitions.terminated,
            value_target=value_target,
            value_mask=value_mask,
        )

    return jax.jit(selfplay)


def flatten_selfplay_data(data: SelfPlayData) -> SelfPlayData:
    """Flatten ``[time, batch, ...]`` self-play data to ``[examples, ...]``."""

    def flatten(array: jax.Array) -> jax.Array:
        return array.reshape((array.shape[0] * array.shape[1], *array.shape[2:]))

    return SelfPlayData(
        observation=flatten(data.observation),
        action_weights=flatten(data.action_weights),
        reward=flatten(data.reward),
        discount=flatten(data.discount),
        terminated=flatten(data.terminated),
        value_target=flatten(data.value_target),
        value_mask=flatten(data.value_mask),
    )


def mirror_selfplay_data(data: SelfPlayData) -> SelfPlayData:
    """Augment flattened Connect Four self-play data with its horizontal mirror.

    Returns a SelfPlayData with twice as many examples: the originals, then
    each example's column-flipped counterpart. Connect Four has horizontal
    symmetry across the centre column, so the mirrored position has the same
    game-theoretic value and the same optimal move distribution (just with
    mirrored column indices). This doubles the effective training set with
    zero extra self-play cost — directly attacks value-signal quality.

    Assumes the input is already flattened to ``[examples, ...]`` (post
    ``flatten_selfplay_data``). The observation axis layout is pgx's
    ``[batch, rows, cols, planes]``; ``action_weights`` is ``[batch, cols]``.
    Reward/discount/terminated/value_target/value_mask are scalar per example
    and column-independent, so they're concatenated unchanged.
    """

    mirrored_observation = data.observation[:, :, ::-1, :]
    mirrored_action_weights = data.action_weights[:, ::-1]
    return SelfPlayData(
        observation=jnp.concatenate([data.observation, mirrored_observation], axis=0),
        action_weights=jnp.concatenate(
            [data.action_weights, mirrored_action_weights], axis=0
        ),
        reward=jnp.concatenate([data.reward, data.reward], axis=0),
        discount=jnp.concatenate([data.discount, data.discount], axis=0),
        terminated=jnp.concatenate([data.terminated, data.terminated], axis=0),
        value_target=jnp.concatenate([data.value_target, data.value_target], axis=0),
        value_mask=jnp.concatenate([data.value_mask, data.value_mask], axis=0),
    )
