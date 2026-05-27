"""JAX + pgx + mctx Connect Four self-play throughput spike.

Measures self-play throughput for a batched, jit-compiled AlphaZero-style
self-play loop (pgx `connect_four` env + `mctx` Gumbel AlphaZero + a vendored
AZNet), to compare head-to-head against the PyTorch process-pool self-play in
the main `alphazero` repo (which logs `self_play_games_per_sec` to wandb).

The self-play structure mirrors pgx's official examples/alphazero/train.py,
reduced to inference-only timing: no gradient updates, no checkpointing, and a
single-device `jax.jit` instead of `jax.pmap` (this box is CPU-only).

IMPORTANT — what this does and does not measure:
  - On CPU, XLA still vmaps the whole batch and fuses the scan, so this captures
    the *vectorization* characteristic of the JAX stack vs a Python loop.
  - It does NOT capture JAX's headline 10-100x, which is a GPU/TPU result. For
    that number, run this same script in a GPU container (e.g. Modal A10/A100).

Usage:
    uv run python bench.py --batch-size 256 --num-simulations 32 \
        --num-channels 64 --num-blocks 5 --max-steps 64 --timed-iters 3
"""

from __future__ import annotations

import argparse
import time

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from network import AZNet

# Connect Four: a complete game is at most 42 plies; 64 scan steps comfortably
# covers a batch of games to termination (auto_reset restarts finished games).
_ENV_ID = "connect_four"


def build_forward(num_actions: int, num_channels: int, num_blocks: int):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=num_actions,
            num_channels=num_channels,
            num_blocks=num_blocks,
            resnet_v2=True,
        )
        return net(x, is_training=not is_eval, test_local_stats=False)

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_selfplay(
    env, forward, *, num_simulations: int, batch_size: int, max_steps: int
):
    """Return a jitted self-play fn that plays `batch_size` games for `max_steps`
    plies and returns the per-step `terminated` mask (used to count games)."""

    def recurrent_fn(model, rng_key, action, state):
        del rng_key
        params, bn_state = model
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        (logits, value), _ = forward.apply(
            params, bn_state, state.observation, is_eval=True
        )
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0 * jnp.ones_like(value))
        out = mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        )
        return out, state

    def selfplay(model, rng_key):
        params, bn_state = model

        def step_fn(state, key):
            key1, key2 = jax.random.split(key)
            (logits, value), _ = forward.apply(
                params, bn_state, state.observation, is_eval=True
            )
            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
            policy_output = mctx.gumbel_muzero_policy(
                params=model,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )
            keys = jax.random.split(key2, batch_size)
            state = jax.vmap(auto_reset(env.step, env.init))(
                state, policy_output.action, keys
            )
            return state, state.terminated

        rng_key, sub = jax.random.split(rng_key)
        state = jax.vmap(env.init)(jax.random.split(sub, batch_size))
        _, terminated = jax.lax.scan(
            step_fn, state, jax.random.split(rng_key, max_steps)
        )
        return terminated  # (max_steps, batch_size) bool

    return jax.jit(selfplay)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-simulations", type=int, default=32)
    p.add_argument("--num-channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--timed-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"jax {jax.__version__} | devices: {jax.devices()}")
    print(
        f"config: batch={args.batch_size} sims={args.num_simulations} "
        f"net={args.num_channels}ch x{args.num_blocks} max_steps={args.max_steps}"
    )

    env = pgx.make(_ENV_ID)
    forward = build_forward(env.num_actions, args.num_channels, args.num_blocks)

    key = jax.random.PRNGKey(args.seed)
    dummy = jax.vmap(env.init)(jax.random.split(key, 2))
    model = forward.init(key, dummy.observation)  # (params, bn_state)

    selfplay = make_selfplay(
        env,
        forward,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    # Warmup: triggers XLA compilation (excluded from timing).
    t0 = time.perf_counter()
    term = selfplay(model, key)
    term.block_until_ready()
    compile_s = time.perf_counter() - t0
    print(f"compile + first run: {compile_s:.1f}s")

    # Timed runs.
    total_s = 0.0
    total_games = 0
    for i in range(args.timed_iters):
        k = jax.random.fold_in(key, i + 1)
        t0 = time.perf_counter()
        term = selfplay(model, k)
        term.block_until_ready()
        dt = time.perf_counter() - t0
        games = int(term.sum())  # completed games (auto_reset terminations)
        total_s += dt
        total_games += games

    env_steps = args.batch_size * args.max_steps * args.timed_iters
    games_per_sec = total_games / total_s
    steps_per_sec = env_steps / total_s
    # Each ply runs `num_simulations` batched env.step calls inside MCTS.
    sims_per_sec = steps_per_sec * args.num_simulations

    print("\n=== throughput (CPU, single device) ===")
    print(f"avg time / self-play call : {total_s / args.timed_iters:.2f}s")
    print(f"completed games           : {total_games} over {args.timed_iters} calls")
    print(f"COMPLETED GAMES / SEC     : {games_per_sec:.1f}")
    print(f"env-steps / sec           : {steps_per_sec:.0f}")
    print(f"MCTS env.step calls / sec : {sims_per_sec:.0f}")


if __name__ == "__main__":
    main()
