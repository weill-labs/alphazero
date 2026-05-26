# AlphaZero (tic-tac-toe)

An implementation of **AlphaGo Zero** in PyTorch: a single two-headed neural network
(policy + value) improved by Monte Carlo Tree Search and self-play reinforcement learning,
with **no human data and no oracle**. The pipeline is game-agnostic; tic-tac-toe is the first
validation target because optimal play is known (perfect play is a draw).

## Result

Trained from **pure self-play**, the agent reaches optimal tic-tac-toe play. Across 5 seeds at
the default budget (`60` iterations × `24` self-play games × `128` MCTS sims), evaluated over 40
games against a minimax (perfect) player:

| metric | result |
| --- | --- |
| vs. perfect player | **0 losses / 40 draws** every seed (optimal) |
| vs. random player | ~37–39 wins / 1–3 draws / 0 losses |

The `PerfectPlayer` (minimax) is used **only for evaluation**, never for training.

## How it works

```
   ┌──────────────────────────────────────────────┐
   ▼                                                │
 self-play (MCTS) ──► (state, π, z) ──► train net ──┘
```

- **Self-play** plays full games using PUCT-guided MCTS at each move, recording
  `(canonical_state, visit-count policy π, outcome z)` per position.
- **Training** fits the policy head to π (cross-entropy) and the value head to z (MSE).
- A stronger network makes the search stronger, which yields better training targets — the
  self-improving loop.

Architecture and module contracts: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

```bash
uv sync                 # core (torch, numpy, wandb)
uv sync --extra dev     # + pytest
uv sync --extra modal   # + modal (for cloud training)
```

To sync every optional dependency used by the local helper commands:

```bash
make sync
```

## Common commands

```bash
make test   # uv run --extra dev pytest
make modal  # uv run --extra modal modal run modal_app.py
make bench  # uv run --extra dev python -m alphazero.benchmark
```

## Train locally

```bash
uv run python -m alphazero.arena            # robust defaults: 60 iters x 24 games x 128 sims
uv run python -m alphazero.arena --no-wandb # disable wandb logging
```

wandb logging is **on by default** (project `alphazero-tictactoe`); the run URL is printed at
start. Per-iteration losses plus timing/throughput (`iteration_seconds`, `iters_per_sec`,
`self_play_games_per_sec`) are logged, and final eval vs. perfect/random. Tests force
`WANDB_MODE=disabled`, so the suite never touches the network.

## Train on Modal (optional, cloud)

```bash
uv run modal setup                                   # one-time auth (browser)
# requires a Modal secret named "wandb" containing WANDB_API_KEY
uv run --extra modal modal run modal_app.py --seed 0
uv run --extra modal modal run modal_app.py --gpu A10G --seed 0   # GPU (see note)
```

Runs training in the cloud without using local resources; local training is unchanged.

> GPU note: tic-tac-toe uses a tiny network and MCTS does sequential single-position
> inference, so a GPU does not help (and can be slower) at this scale. GPUs pay off with
> batched MCTS and larger networks/games.

## Connect Four

Connect Four is supported as a larger game-agnostic validation target. The board has
`7` columns by `6` rows, moves are gravity drops into a column, and a player wins by
making `4` in a row horizontally, vertically, or diagonally.

Select the game with `--game {tictactoe,connectfour}`. Tic-tac-toe remains the default.

```bash
uv run python -m alphazero.arena --game connectfour
uv run --extra modal modal run modal_app.py --game connectfour
```

Connect Four does not have a tractable perfect oracle in this project, so evaluation uses
practical checks instead:

- **Tactical correctness:** never miss an immediate win and always block an immediate loss.
- **Baseline strength:** win-rate against `RandomPlayer`.
- **Search baseline:** games against a negamax/alpha-beta baseline opponent.
- **Human inspection:** human-vs-agent play through `alphazero/play.py`.

## Test

```bash
uv run --extra dev pytest
```

## Layout

```
alphazero/
  game.py            # abstract Game interface
  games/tictactoe.py # tic-tac-toe rules
  network.py         # two-headed net (policy + value)
  mcts.py            # PUCT search
  selfplay.py        # self-play data generation
  train.py           # training loop + loss
  arena.py           # players, evaluation, training entrypoint
modal_app.py         # optional Modal cloud-training app
docs/ARCHITECTURE.md # integration contract
```
