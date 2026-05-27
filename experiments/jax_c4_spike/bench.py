"""JAX + pgx + mctx Connect Four self-play throughput spike.

Measures self-play throughput for a batched, jit-compiled AlphaZero-style
self-play loop (pgx `connect_four` env + `mctx` Gumbel AlphaZero + a vendored
AZNet), to compare head-to-head against the PyTorch process-pool self-play in
the main `alphazero` repo (which logs `self_play_games_per_sec` to wandb).

The self-play structure mirrors pgx's official examples/alphazero/train.py,
reduced to inference-only timing: no gradient updates, no checkpointing, and a
single-device `jax.jit` instead of `jax.pmap`.

`run_benchmark` is the reusable core (returns a metrics dict); `main` is the
CLI. The same `run_benchmark` is invoked on a Modal GPU by `modal_bench.py` so
the CPU and GPU numbers come from identical code.

Usage:
    uv run python bench.py --batch-size 256 --num-simulations 32 \
        --num-channels 64 --num-blocks 5 --max-steps 64 --timed-iters 3
"""

from __future__ import annotations

import argparse
import contextlib
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


def run_benchmark(
    *,
    batch_size: int = 256,
    num_simulations: int = 32,
    num_channels: int = 64,
    num_blocks: int = 5,
    max_steps: int = 64,
    timed_iters: int = 3,
    seed: int = 0,
    profile_dir: str | None = None,
) -> dict[str, object]:
    """Time batched self-play and return throughput metrics.

    Compilation (the first call) is excluded from the timed average. Completed
    games are counted from the per-step termination mask, so games/sec is
    directly comparable to the PyTorch `self_play_games_per_sec`.

    When ``profile_dir`` is set, the timed runs (post-compile, steady state) are
    wrapped in a ``jax.profiler.trace`` so the XLA op-level breakdown is captured
    for TensorBoard/Perfetto.
    """
    env = pgx.make(_ENV_ID)
    forward = build_forward(env.num_actions, num_channels, num_blocks)

    key = jax.random.PRNGKey(seed)
    dummy = jax.vmap(env.init)(jax.random.split(key, 2))
    model = forward.init(key, dummy.observation)  # (params, bn_state)

    selfplay = make_selfplay(
        env,
        forward,
        num_simulations=num_simulations,
        batch_size=batch_size,
        max_steps=max_steps,
    )

    # Warmup triggers XLA compilation (excluded from timing).
    t0 = time.perf_counter()
    term = selfplay(model, key)
    term.block_until_ready()
    compile_s = time.perf_counter() - t0

    total_s = 0.0
    total_games = 0
    profile_ctx = (
        jax.profiler.trace(profile_dir) if profile_dir else contextlib.nullcontext()
    )
    with profile_ctx:
        for i in range(timed_iters):
            k = jax.random.fold_in(key, i + 1)
            t0 = time.perf_counter()
            term = selfplay(model, k)
            term.block_until_ready()
            total_s += time.perf_counter() - t0
            total_games += int(term.sum())  # completed games (auto_reset terminations)

    env_steps = batch_size * max_steps * timed_iters
    steps_per_sec = env_steps / total_s
    return {
        "jax_version": jax.__version__,
        "devices": str(jax.devices()),
        "config": {
            "batch_size": batch_size,
            "num_simulations": num_simulations,
            "num_channels": num_channels,
            "num_blocks": num_blocks,
            "max_steps": max_steps,
            "timed_iters": timed_iters,
        },
        "compile_s": round(compile_s, 2),
        "avg_call_s": round(total_s / timed_iters, 3),
        "completed_games": total_games,
        "games_per_sec": round(total_games / total_s, 2),
        "env_steps_per_sec": round(steps_per_sec, 1),
        "mcts_env_steps_per_sec": round(steps_per_sec * num_simulations, 1),
        "profile_dir": profile_dir,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-simulations", type=int, default=32)
    p.add_argument("--num-channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--timed-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--profile",
        action="store_true",
        help="capture a jax.profiler trace of the timed runs (XLA op breakdown)",
    )
    p.add_argument(
        "--profile-dir",
        default="jax_trace",
        help="directory for the profiler trace (with --profile)",
    )
    args = p.parse_args()

    result = run_benchmark(
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        max_steps=args.max_steps,
        timed_iters=args.timed_iters,
        seed=args.seed,
        profile_dir=args.profile_dir if args.profile else None,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    if result.get("profile_dir"):
        print(
            f"\ntrace written to {result['profile_dir']}/ — view with:\n"
            f"  tensorboard --logdir {result['profile_dir']}\n"
            "(needs: pip install tensorboard tensorboard-plugin-profile)"
        )


if __name__ == "__main__":
    main()
