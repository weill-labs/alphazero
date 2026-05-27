# JAX migration — pure-JAX AlphaZero, Connect Four first

**Status:** in progress. Tracking epic: `alphago-6ud`.

Replace the PyTorch implementation with a JAX stack and drive Connect Four to
*solved* (blunder-rate 0 against the exact `c4_solver` oracle). Connect Four is
the pilot: it is small (fast iteration) and it has an exact oracle, so we can
prove the new stack is *correct*, not merely fast. The CPU vs GPU benchmark
(`experiments/jax_c4_spike/`, PR #33) showed the JAX self-play stack reaches
~10–24× the PyTorch throughput on a GPU; that throughput is what should push C4
past the plateau the PyTorch campaign hit (blunder-rate stuck at 0.125).

## Decision log

- **Big-bang, no dual stack.** We do not keep PyTorch as a parallel baseline.
  Execution order is build-JAX-then-remove-torch (Phase 4), so the repo is never
  bricked mid-flight, but the end state is torch-free.
- **No backwards compatibility.** No compat shims, no migration helpers, no
  keeping old code "just in case." Delete cleanly.
- **Net library: Flax NNX** (`flax.nnx`), the current Flax API.
- **Environment: `pgx` `connect_four`** (proven, GPU-vectorized, already used in
  the benchmark). We bridge its board state to the solver rather than porting a
  new env.
- **MCTS / self-play: `mctx`** (Gumbel AlphaZero), batched on-device.
- **Optimizer: `optax`** (adam).
- **Replay buffer: start buffer-free** — train on each iteration's fresh
  self-play data (matches the pgx example). Add a buffer only if C4 will not
  solve without one.
- **Certification: faithful** — certify the agent's *MCTS* move (a few `mctx`
  sims), not raw-policy argmax, since that is how the agent actually plays.

## Stack & layout

New top-level package `jaxzero/` (Python 3.12, `uv`). Built on the mctx + pgx +
`lax.scan` self-play structure already prototyped in
`experiments/jax_c4_spike/bench.py` — reuse that structure; swap the Haiku net
for a Flax NNX net.

```
jaxzero/
  net.py         # Flax NNX AlphaZero residual net
  selfplay.py    # pgx connect_four + mctx self-play -> training data
  train.py       # optax training loop + checkpoint save/load
  cli.py         # `python -m jaxzero.train --iterations ...`
tests/           # hermetic smoke + determinism tests for the above
```

## Keep / refactor / remove

- **Keep unchanged (framework-agnostic):** `alphazero/c4_solver.py` (exact
  oracle — the definition of "solved"), `alphazero/elo_ladder.py` (pure Elo
  math).
- **Refactor (Phase 2):** `alphazero/c4_certify.py` — decouple from the torch
  `MCTSPlayer`/`load_checkpoint`; express it against a framework-agnostic
  `Agent` protocol so a JAX agent plugs in.
- **Remove (Phase 4, with explicit approval):** the six torch files —
  `alphazero/{network,train,arena,play,benchmark}.py` and the torch path in
  `modal_app.py`; plus `alphazero/{mcts,selfplay}.py` if fully superseded.

## Phases

### Phase 1 — JAX C4 training pipeline (`alphago-vlr`)
Build `jaxzero/`:
- **net.py**: Flax NNX residual net. Input = pgx `connect_four` observation
  (planes, H, W); outputs `(policy_logits[action_size], value)` with
  `tanh` value. Configurable `channels`, `num_res_blocks`.
- **selfplay.py**: `mctx.gumbel_muzero_policy` self-play over a vmapped batch of
  pgx games via `lax.scan` (mirror `bench.py`), collecting per-step
  `(obs, action_weights, reward, discount, terminated)`; compute value targets
  by discounted return (mask truncated episodes).
- **train.py**: optax adam loop — loss = softmax-cross-entropy(policy_logits,
  action_weights) + MSE(value, value_target) (mask truncated); checkpoint
  save/load of NNX params (orbax or msgpack). Buffer-free.
- **cli.py**: `python -m jaxzero.train --iterations N --batch-size B --sims S ...`.
- **Tests:** a fast hermetic smoke test (few iters, tiny net, tiny batch) and a
  determinism test (same seed → same params). No torch imported anywhere.

### Phase 2 — certify bridge (`alphago-h7q`, blocked by Phase 1)
- Refactor `c4_certify.py` to an `Agent` protocol (`move(state)`,
  `value(state)`), independent of torch.
- Adapter: pgx `connect_four` board state ↔ `c4_solver.ConnectFourState`.
- JAX agent = Phase-1 net + `mctx` implements the protocol.
- Certify a checkpoint against `c4_solver`: blunder-rate, policy-match,
  value-MAE, single solved verdict. `c4_solver.py` stays untouched.

### Phase 3 — JAX Modal GPU entrypoint (`alphago-rc8`, blocked by Phase 1)
- New JAX Modal app: `jax[cuda12]` image, GPU, runs the Phase-1 training on GPU.
- Persist checkpoints to the existing `alphazero-checkpoints` Volume, reusing the
  `/checkpoints/<run_tag>/<game>/` scheme; log to the per-game wandb project.

### Phase 4 — remove PyTorch (`alphago-bdi`, blocked by Phases 1–3)
- Delete the torch modules and torch deps; add `jax/pgx/mctx/flax/optax`; update
  `__init__`/tests; repo lands torch-free. **Requires explicit deletion approval
  (RULE 1).**

## Contracts (so phases integrate)

- **Net forward:** `net(obs_batch) -> (policy_logits[B, action_size], value[B])`.
- **Checkpoint:** a single file storing the NNX param state + the net config
  (`channels`, `num_res_blocks`, `action_size`, obs shape) so it reloads without
  guessing dims.
- **Agent protocol (Phase 2):** `move(board) -> int`, `value(board) -> float`,
  operating on the `c4_solver` board representation via the adapter.

## Working agreements for delegated workers

- Build only your phase; follow this doc for interfaces.
- **Code-only commits** — do not touch `.beads/` (the Dolt DB is not in your
  worktree). Open a PR; the orchestrator updates beads centrally.
- Local gate (no CI): `pytest`, `ruff check`, `ruff format`, `ubs` must pass on
  changed files before the PR.
- Determinism: seed everything; a fixed seed must reproduce params/results.
