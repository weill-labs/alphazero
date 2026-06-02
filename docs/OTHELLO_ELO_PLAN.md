# Othello / Elo Plan

## Why Othello first

C4 should now be treated as a regression benchmark, not the main optimization
target. The next-game path should test whether the Tier-1 transformer is more
useful on a larger board where local and long-range structure matter more.

Use Othello first:

- Local `pgx` supports `othello`.
- Initial observation shape is `(8, 8, 2)`.
- Action size is `65` (`64` board moves plus pass).
- It is larger than C4 but still small enough for fast iteration.
- `pgx` in this environment does not expose Gomoku, so Gomoku would require
  adding a rules/env layer before learning anything.

Do not carry C4 solver expectations forward. Othello needs non-solver
evaluation.

## Current hardwired seams

The network can already handle arbitrary observation shape and action size via
`AlphaZeroNetConfig`. The main C4-specific seams are elsewhere:

- `jaxzero.selfplay`: `ENV_ID = "connect_four"` and `make_env()` has no game
  parameter.
- `jaxzero.evaluate`: vs-random evaluator is fixed to Connect Four.
- `jaxzero.arena`: gating match is fixed to Connect Four.
- `jaxzero.cli`: no `--game`; C4 solver eval is wired by default when
  `--solver-eval-positions > 0`.
- `jaxzero.modal_train`: validates only `connectfour`.
- C4-only features (`--mirror-augment`, `--solver-rehearsal-*`,
  `--policy-head-style per_column`) should not silently run on Othello.

## Minimal implementation path

1. Add a small pgx game spec layer.

   Keep this intentionally boring: map user-facing names to pgx env IDs and a
   few capability flags.

   - `connectfour` -> `connect_four`, solver eval yes, mirror yes, default
     max_steps `64`.
   - `othello` -> `othello`, solver eval no, mirror initially no, default
     max_steps `128`.

2. Thread the game spec through JAX training.

   - Add `game: str = "connectfour"` to `TrainingConfig`.
   - Pass `game`/`env_id` into `make_selfplay`, `make_evaluator`, and
     `make_gating_match`.
   - Use `initial_observation_shape(game)` when creating new checkpoints.
   - Keep checkpoint loading shape-driven as it is today.

3. Make C4-only flags explicit.

   - Reject solver rehearsal unless `game == "connectfour"`.
   - Reject C4 solver eval unless `game == "connectfour"`.
   - Reject `--mirror-augment` on Othello until an action/observation symmetry
     transform is implemented and tested.
   - Reject `--policy-head-style per_column` on Othello; Othello has 65 actions
     and does not satisfy the C4 column-policy assumption.

4. Add Othello smoke tests.

   - `make_selfplay(..., game="othello")` returns observations shaped
     `[time, batch, 8, 8, 2]` and action weights shaped `[time, batch, 65]`.
   - `run_training(TrainingConfig(game="othello", iterations=1, ...))` returns
     metrics and can checkpoint/load.
   - vs-random and gating run with balanced seats.
   - C4-only flags fail clearly on Othello.

5. Build non-solver checkpoint evaluation.

   The first Othello evaluator should be checkpoint-vs-checkpoint Elo, not
   exact blunder rate.

   - Balanced seats.
   - Greedy policy baseline first for speed.
   - Optional MCTS eval mode after the plumbing is stable.
   - Checkpoint directory ladder support so we can select best periodic
     checkpoint instead of trusting `final.msgpack`.
   - Report Elo, score, win/draw/loss, games per pairing, and evaluator mode.

6. Run the actual A/B only after the evaluator works.

   Compare matched ResNet vs Tier-1 transformer on Othello:

   - Same self-play batch, sims, iterations, seeds, and checkpoint cadence.
   - At least 3 seeds before claiming architecture superiority.
   - Select best periodic checkpoint by Othello Elo ladder.
   - C4 remains a regression benchmark, not the target metric.

## Stop conditions

Do not start Modal training until local Othello one-iteration smoke and
checkpoint-vs-checkpoint Elo smoke both pass.

Do not claim an Othello architecture result from final checkpoints only. Use the
checkpoint ladder, just like C4.
