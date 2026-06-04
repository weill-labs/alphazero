# Othello Architecture A/B Runs

## 2026-06-02 launch

Bead: `alphago-iaf`

Goal: compare ResNet vs board-aware transformer on Othello using matched seeds
and best-periodic-checkpoint Elo, not final-only metrics.

Shared training config:

- `game=othello`
- `iterations=80`
- `batch_size=128`
- `num_simulations=64`
- `max_steps=128` via Othello default
- `checkpoint_every=20`
- `eval_interval=20`
- `eval_games=64`
- `replay_capacity=65536`
- `minibatch_size=2048`
- `learning_rate=0.001`
- `solver_eval_positions=0`
- Modal GPU requested through `main --gpu A100-40GB`

Architecture configs:

- ResNet: `channels=128`, `num_res_blocks=5`
- Transformer: `arch=transformer`, `d_model=128`, `num_layers=6`,
  `num_heads=4`, `mlp_dim=512`, `use_value_cls_token=true`,
  `input_embed_style=conv3x3`, `policy_head_style=flatten`

Launch notes:

- The first detached attempts through `setsid ... modal run --detach ...::main`
  without backgrounding created idle apps with zero tasks. They did not create
  checkpoints and are not part of the A/B.
- Two tiny Modal smoke runs succeeded:
  - `/checkpoints/othello-modal-smoke/othello/`
  - `/checkpoints/othello-detach-smoke/othello/`
- `--spawn` was tested but an ephemeral app stopped before useful training
  persisted, so the production launches used the known C4 pattern: backgrounded
  `setsid` Modal clients with `--detach`, one per run, logging to `/tmp`.

## Active Runs

| Run tag | Arch | Seed | Modal app | W&B run | Checkpoint dir | Local log |
| --- | --- | ---: | --- | --- | --- | --- |
| `othello-resnet-s101` | resnet | 101 | `ap-79Ew15eb25Jqi5ZBsvkN7T` | `m7rxr2zr` | `/checkpoints/othello-resnet-s101/othello/` | `/tmp/othello-resnet-s101.modal.log` |
| `othello-resnet-s102` | resnet | 102 | `ap-4sFTAHk3IxKK2K8MzzW7Fd` | `tvmo8k2s` | `/checkpoints/othello-resnet-s102/othello/` | `/tmp/othello-resnet-s102.modal.log` |
| `othello-resnet-s103` | resnet | 103 | `ap-zJYwtBN4KGJV1xl0Nwfasz` | `4wla5hzt` | `/checkpoints/othello-resnet-s103/othello/` | `/tmp/othello-resnet-s103.modal.log` |
| `othello-transformer-s101` | transformer | 101 | `ap-W3x7UNoP5Tas9bYHTtDk3N` | `bfnvstdj` | `/checkpoints/othello-transformer-s101/othello/` | `/tmp/othello-transformer-s101.modal.log` |
| `othello-transformer-s102` | transformer | 102 | `ap-kt4kEP3EsnnXS1wOV3HvDj` | `riiltodo` | `/checkpoints/othello-transformer-s102/othello/` | `/tmp/othello-transformer-s102.modal.log` |
| `othello-transformer-s103` | transformer | 103 | `ap-owvHgKcXsSKDqKMy6mcH2r` | `og2ljhjt` | `/checkpoints/othello-transformer-s103/othello/` | `/tmp/othello-transformer-s103.modal.log` |

## Monitoring

```bash
uv run --extra modal modal app list
tail -40 /tmp/othello-resnet-s101.modal.log
uv run --extra modal modal volume ls alphazero-checkpoints /othello-resnet-s101/othello
```

## Elo Evaluation

After a run has `iter_0020.msgpack`, `iter_0040.msgpack`,
`iter_0060.msgpack`, `iter_0080.msgpack`, and `final.msgpack`, rank that run:

```bash
uv run jaxzero-checkpoint-elo --game othello \
  --checkpoint-dir checkpoints/<run-tag>/othello \
  --games-per-pairing 8
```

For local evaluation, first download the run's checkpoint directory from the
Modal volume into `checkpoints/<run-tag>/othello`.

## Pending

- Record wall-clock runtime and Modal cost once jobs finish.
- Download checkpoint ladders.
- Run per-seed checkpoint Elo.
- Compare best-checkpoint Elo distributions across the three matched seeds.

## 2026-06-02 17:14 UTC Status

All six Modal apps were still active with `Tasks=1`.

W&B summary iterations:

- `othello-resnet-s101`: 37
- `othello-resnet-s102`: 23
- `othello-resnet-s103`: 39
- `othello-transformer-s101`: 18
- `othello-transformer-s102`: 19
- `othello-transformer-s103`: 18

Checkpoint volume status:

- All six runs had `iter_0020.msgpack`.
- `othello-resnet-s103` also had `iter_0040.msgpack`.

Local checkpoint downloads completed for all six `iter_0020` files and
`othello-resnet-s103/iter_0040.msgpack`. The downloaded files are under
`checkpoints/<run-tag>/othello/` and are ignored by git.

Evaluator smoke:

```bash
uv run jaxzero-checkpoint-elo --game othello --mode round-robin \
  --games-per-pairing 2 --max-steps 128 --fit-iterations 20 \
  checkpoints/othello-resnet-s101/othello/iter_0020.msgpack \
  checkpoints/othello-resnet-s102/othello/iter_0020.msgpack \
  checkpoints/othello-resnet-s103/othello/iter_0020.msgpack \
  checkpoints/othello-transformer-s101/othello/iter_0020.msgpack \
  checkpoints/othello-transformer-s102/othello/iter_0020.msgpack \
  checkpoints/othello-transformer-s103/othello/iter_0020.msgpack
```

The smoke completed successfully. Treat the numbers as a pipeline check only:
2 games per pairing is too small for an architecture claim.

## 2026-06-02 17:31 UTC Preliminary Greedy Elo

Additional checkpoints downloaded locally:

- `othello-resnet-s101`: `iter_0040`, `iter_0060`, `iter_0080`, `final`
- `othello-resnet-s102`: `iter_0040`
- `othello-resnet-s103`: `iter_0060`, `iter_0080`, `final`
- all three transformer seeds: `iter_0040`

Live status at the time of this checkpoint:

- `othello-resnet-s101`: finished at training summary iteration 79
- `othello-resnet-s102`: running at summary iteration 47
- `othello-resnet-s103`: finished at training summary iteration 79
- transformer seeds: running at summary iterations 38, 40, 38

Greedy Elo diagnostics used `--games-per-pairing 16 --max-steps 128` and two
evaluator seeds where noted. These are not final architecture results because
they are greedy-policy matches and the transformer checkpoint ladders are not
complete yet.

Completed ResNet ladder diagnostics:

- `othello-resnet-s101`: seed 0 best was `iter_0040` (+35.9 Elo vs
  `iter_0020` anchor); seed 1 best was `iter_0020`. Later checkpoints were
  consistently below the early checkpoints.
- `othello-resnet-s103`: seed 0 and seed 1 both selected `final` as best
  (+274.2 and +266.1 Elo vs `iter_0020` anchor). `iter_0080` was second and
  `iter_0060` was the trough in both runs.

Matched `iter_0040` six-model round-robin, two evaluator seeds:

| Checkpoint | Seed 0 Elo | Seed 1 Elo |
| --- | ---: | ---: |
| `othello-resnet-s101/iter_0040` | 0.0 | 0.0 |
| `othello-resnet-s102/iter_0040` | 174.9 | 191.4 |
| `othello-resnet-s103/iter_0040` | -56.4 | -71.2 |
| `othello-transformer-s101/iter_0040` | 270.8 | 274.9 |
| `othello-transformer-s102/iter_0040` | 991.1 | 952.2 |
| `othello-transformer-s103/iter_0040` | 909.5 | 952.2 |

Early read: at equal `iter_0040`, the transformer seeds are ahead in greedy
policy play, with seeds 102 and 103 far ahead. Do not claim the A/B yet; the
next required comparison is best-checkpoint Elo after `60/80/final` checkpoints
exist for all transformer and ResNet seeds.

## 2026-06-02 17:40 UTC ResNet Ladders Complete

All three ResNet runs finished and their full checkpoint ladders are local.
Transformer runs were still active at summary iterations 65, 67, and 65.

Full ResNet greedy-ladder best checkpoints, using
`--games-per-pairing 16 --max-steps 128`:

| Run | Seed 0 best | Seed 0 Elo | Seed 1 best | Seed 1 Elo |
| --- | --- | ---: | --- | ---: |
| `othello-resnet-s101` | `iter_0040` | 35.9 | `iter_0020` | 0.0 |
| `othello-resnet-s102` | `final` | 808.8 | `iter_0080` | 838.1 |
| `othello-resnet-s103` | `final` | 274.2 | `final` | 266.1 |

Matched `iter_0060` six-model round-robin, two evaluator seeds:

| Checkpoint | Seed 0 Elo | Seed 1 Elo |
| --- | ---: | ---: |
| `othello-resnet-s101/iter_0060` | 0.0 | 0.0 |
| `othello-resnet-s102/iter_0060` | 706.2 | 697.5 |
| `othello-resnet-s103/iter_0060` | 610.3 | 650.5 |
| `othello-transformer-s101/iter_0060` | 1277.5 | 1128.9 |
| `othello-transformer-s102/iter_0060` | 947.8 | 960.4 |
| `othello-transformer-s103/iter_0060` | 1035.0 | 968.3 |

Early read: the transformer lead persists at equal `iter_0060`, but this is
still not the final A/B. The final comparison must use best-checkpoint ladders
for the transformer seeds once `iter_0080` and `final` exist.

## 2026-06-02 17:55 UTC Final Greedy Checkpoint A/B

All six production runs finished at summary iteration 79 and all checkpoint
ladders were downloaded locally.

Cost proxy from W&B runtime, assuming one A100-class GPU per run:

| Run | Runtime seconds | Runtime hours |
| --- | ---: | ---: |
| `othello-resnet-s101` | 1321.7 | 0.367 |
| `othello-resnet-s102` | 2187.6 | 0.608 |
| `othello-resnet-s103` | 1291.2 | 0.359 |
| `othello-transformer-s101` | 2689.1 | 0.747 |
| `othello-transformer-s102` | 2625.2 | 0.729 |
| `othello-transformer-s103` | 2686.0 | 0.746 |

Total runtime proxy: 12,800.8 seconds, or 3.556 A100-job-hours. ResNets used
1.333 job-hours; transformers used 2.222 job-hours.

Best checkpoints were selected by averaging the two per-run greedy ladder
evaluator seeds:

| Run | Selected checkpoint | Ladder seed 0 Elo | Ladder seed 1 Elo |
| --- | --- | ---: | ---: |
| `othello-resnet-s101` | `iter_0040` | 35.9 | -6.2 |
| `othello-resnet-s102` | `iter_0080` | 786.7 | 838.1 |
| `othello-resnet-s103` | `final` | 274.2 | 266.1 |
| `othello-transformer-s101` | `iter_0060` | 414.3 | 530.3 |
| `othello-transformer-s102` | `iter_0060` | 365.8 | 388.9 |
| `othello-transformer-s103` | `final` | 505.9 | 481.9 |

Best-checkpoint six-model round-robin used `--games-per-pairing 32`,
`--fit-iterations 300`, and evaluator seeds 0, 1, and 2:

| Best checkpoint | Seed 0 Elo | Seed 1 Elo | Seed 2 Elo | Mean Elo |
| --- | ---: | ---: | ---: | ---: |
| `othello-resnet-s101/iter_0040` | 0.0 | 0.0 | 0.0 | 0.0 |
| `othello-resnet-s102/iter_0080` | 353.0 | 390.9 | 368.5 | 370.8 |
| `othello-resnet-s103/final` | -130.6 | -17.9 | -47.0 | -65.2 |
| `othello-transformer-s101/iter_0060` | 318.1 | 433.7 | 409.0 | 386.9 |
| `othello-transformer-s102/iter_0060` | 341.1 | 415.8 | 365.4 | 374.1 |
| `othello-transformer-s103/final` | 380.5 | 390.1 | 381.6 | 384.1 |

Architecture read under greedy Elo: transformer wins the distribution. All
three transformer seeds cluster around 374-387 mean Elo; only one ResNet seed
(`s102`) is competitive, at 370.8 mean Elo. The mean across all seed/evaluator
points is 381.7 for transformer versus 101.9 for ResNet. This is a real signal
for Othello, but the best individual ResNet is close enough to the transformer
cluster that the next decision should use a stronger MCTS-style evaluator before
locking architecture changes.

Failure modes observed:

- `--spawn` on an ephemeral Modal app returned a call id but did not persist a
  useful training run.
- Detached attempts that were not backgrounded correctly created stopped/idle
  apps with zero tasks.
- Production runs using backgrounded `setsid ... modal run --detach` stayed up
  and produced complete checkpoint ladders.

## 2026-06-03 05:17 UTC MCTS Verification

Bead: `alphago-jvj`

Added `jaxzero-checkpoint-elo --evaluator-mode mcts`, backed by deterministic
`mctx.gumbel_muzero_policy` with `--gumbel-scale 0.0`. Focused tests passed:

```bash
uv run --extra dev pytest tests/test_jaxzero_checkpoint_elo.py
```

The MCTS evaluator is materially slower than greedy Elo. A 6-model selected
best-checkpoint round-robin at `--mcts-simulations 16`,
`--games-per-pairing 8`, and evaluator seed 0 took roughly 12 minutes locally.

Selected best-checkpoint MCTS round-robin:

| Best checkpoint | MCTS Elo |
| --- | ---: |
| `othello-resnet-s101/iter_0040` | 0.0 |
| `othello-resnet-s102/iter_0080` | 135.1 |
| `othello-resnet-s103/final` | -138.4 |
| `othello-transformer-s101/iter_0060` | 205.1 |
| `othello-transformer-s102/iter_0060` | 762.7 |
| `othello-transformer-s103/final` | 709.4 |

Top-contender confirmation used the best ResNet versus the two strongest
transformers at `--mcts-simulations 16`, `--games-per-pairing 16`, and
evaluator seed 1:

| Checkpoint | MCTS Elo |
| --- | ---: |
| `othello-resnet-s102/iter_0080` | 0.0 |
| `othello-transformer-s102/iter_0060` | 740.6 |
| `othello-transformer-s103/final` | 783.2 |

Pairing detail: `othello-resnet-s102/iter_0080` lost 0-16 to
`othello-transformer-s102/iter_0060` and 0-16 to
`othello-transformer-s103/final`.

MCTS verdict: the greedy transformer edge survives search-backed play. The
best individual ResNet seed that looked close under greedy Elo is not close
under this 16-sim MCTS check. Remaining caveat: this is not a high-replication
MCTS Elo study because the naive local MCTS evaluator is slow; further MCTS work
should run on Modal or optimize pair evaluation before increasing seeds/games.

## 2026-06-03 Default Direction

Bead: `alphago-yn0`

Decision: use the tested Othello transformer preset as the default Othello
architecture direction. User-facing training entrypoints now resolve
`--game othello` with no explicit architecture flags to:

- `arch=transformer`
- `d_model=128`, `num_layers=6`, `num_heads=4`, `mlp_dim=512`
- `use_value_cls_token=true`
- `input_embed_style=conv3x3`
- `policy_head_style=flatten`

Connect Four defaults remain ResNet/v1-compatible. Explicit Othello ResNet
overrides are still supported, but future Othello training should treat ResNet
as a control, not the default path.

## 2026-06-03 Modal MCTS Runner

Bead: `alphago-6tz`

`jaxzero/modal_train.py` now has a `checkpoint_elo` Modal local entrypoint for
running checkpoint Elo directly against the `alphazero-checkpoints` volume. It
defaults to `A100-40GB` and avoids downloading checkpoint ladders for higher-rep
MCTS checks.

Top-contender Othello command:

```bash
uv run --extra modal modal run jaxzero/modal_train.py::checkpoint_elo \
  --game othello \
  --checkpoints "othello-resnet-s102/othello/iter_0080.msgpack,othello-transformer-s102/othello/iter_0060.msgpack,othello-transformer-s103/othello/final.msgpack" \
  --mode round-robin \
  --evaluator-mode mcts \
  --mcts-simulations 32 \
  --games-per-pairing 32 \
  --fit-iterations 300 \
  --seed 2
```

Use `--spawn` for long runs if the local client should detach after submitting
the remote A100 job.

Validation run:

- Modal app: `ap-fQArE2UxC02fGk5BWtS3Vl`
- Exact prior top-contender setup: `--mcts-simulations 16`,
  `--games-per-pairing 16`, `--seed 1`
- Runtime: 63.5 seconds for 48 games on `A100-40GB`
- Result reproduced the local 16-sim check: ResNet s102 lost 0-16 to
  transformer s102 and 0-16 to transformer s103.

32-sim top-contender replication:

| Evaluator seed | ResNet s102 Elo | Transformer s102 Elo | Transformer s103 Elo | Runtime seconds |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.0 | -98.0 | -212.5 | 83.1 |
| 3 | 0.0 | -83.0 | -232.6 | 82.0 |
| 4 | 0.0 | -148.8 | -241.9 | 82.5 |

Aggregate 32-sim pair scores across seeds 2-4:

| Pairing | Score |
| --- | ---: |
| ResNet s102 vs transformer s102 | 42-54 |
| ResNet s102 vs transformer s103 | 96-0 |
| Transformer s102 vs transformer s103 | 44-52 |

Read: the Modal runner is correct, but the Othello MCTS verdict is not stable
across search budgets. At 16 sims, both top transformers crush the best ResNet.
At 32 sims, the best ResNet wins the aggregate round-robin because it sweeps
transformer s103, despite losing or tying head-to-head against transformer s102.
Do not treat Othello architecture as settled until the 16-vs-32-sim flip is
explained with a sim-budget sweep and/or action-level traces.

## 2026-06-04 MCTS Budget Flip Trace

Bead: `alphago-76r`

Controlled top-contender sweep:

- 16 sims, `--games-per-pairing 16`, evaluator seeds 1-4: ResNet s102 lost
  every played game against both transformer s102 and transformer s103
  (`0-64` aggregate against each transformer).
- 24 sims, `--games-per-pairing 32`, evaluator seeds 1-4: transformer s103
  swept ResNet s102 (`128-0`), while ResNet s102 narrowly beat transformer
  s102 (`68-60`).
- 32 sims, `--games-per-pairing 32`, evaluator seeds 1-4: ResNet s102 swept
  transformer s103 (`128-0`) and narrowly lost to transformer s102 (`58-70`).

The flip is therefore a deterministic search-budget bifurcation, not ordinary
match seed noise. The clearest case is ResNet s102 `iter_0080` versus
transformer s103 `final` with pair seed `1099128568`:

| Sims | Pairing result | First selected-action difference |
| ---: | --- | --- |
| 24 | ResNet loses `0-32` | ply 10: ResNet lanes select action `38` |
| 32 | ResNet wins `32-0` | ply 10: ResNet lanes select action `46` |

For that exact pairing seed, selected actions are identical through plies 0-9.
At ply 10, the transformer-controlled lanes still select action `60` in both
runs, but the ResNet-controlled lanes switch from action `38` at 24 sims to
action `46` at 32 sims. The later game branch then flips the full 32-game
pairing.

Trace commands used the Modal checkpoint runner:

```bash
uv run --extra modal modal run jaxzero/modal_train.py::checkpoint_elo \
  --game othello \
  --checkpoints "othello-resnet-s102/othello/iter_0080.msgpack,othello-transformer-s103/othello/final.msgpack" \
  --games-per-pairing 32 \
  --evaluator-mode mcts \
  --mcts-simulations <24-or-32> \
  --seed 1099128568 \
  --trace-plies 20 \
  --trace-summary-only
```

Modal apps: `ap-n2lcahVMh4wsjC2dUN9XDI` for 24 sims and
`ap-DQssP8vfKA7F6G6oEbDlq7` for 32 sims.

Implication: fixed low MCTS budgets are not a neutral architecture comparator
for these Othello checkpoints. The next evaluator should either use much higher
budgets with uncertainty bars or compare models by policy/value calibration on
the traced divergent states before treating MCTS Elo as architecture truth.

## 2026-06-04 Ply-10 Probe Calibration

Bead: `alphago-8bo`

Added `jaxzero-checkpoint-elo --probe-ply` and the same Modal runner plumbing to
replay a two-checkpoint match to a target ply, then probe both checkpoints'
root MCTS choices at several simulation budgets. The probe uses
`PolicyOutput.action` as the selected move because that is what the Elo evaluator
plays. `PolicyOutput.action_weights` are also emitted, but they are MCTS policy
training targets, not necessarily the selected move.

Probe target:

- Pairing: ResNet s102 `iter_0080` vs transformer s103 `final`
- Seed: `1099128568`
- Replay budget: 24 sims
- Target ply: 10
- Games: 32

Selected action by budget at the exact ply-10 state:

| Probe sims | ResNet-controlled lanes | Transformer-controlled lanes |
| ---: | --- | --- |
| 16 | `38` in 17 lanes | `60` in 15 lanes |
| 24 | `38` in 17 lanes | `60` in 15 lanes |
| 32 | `46` in 17 lanes | `60` in 15 lanes |
| 64 | `46` in 17 lanes | `60` in 15 lanes |
| 128 | `39` in 17 lanes | `60` in 15 lanes |
| 256 | `39` in 17 lanes | `34` in 15 lanes |
| 512 | `38` in 17 lanes | `34` in 15 lanes |
| 1024 | `39` in 17 lanes | `34` in 15 lanes |

Representative ResNet lane 2 action weights show why this state is a poor
low-budget evaluator:

| Probe sims | Selected action | Top action weights |
| ---: | ---: | --- |
| 24 | `38` | `38`: 0.489, `39`: 0.253, `46`: 0.248 |
| 32 | `46` | `46`: 0.666, `39`: 0.227, `38`: 0.100 |
| 64 | `46` | `46`: 0.583, `39`: 0.365, `38`: 0.041 |
| 128 | `39` | `38`: 0.660, `23`: 0.149, `54`: 0.112 |
| 256 | `39` | `39`: 0.670, `23`: 0.158, `38`: 0.113 |
| 512 | `38` | `54`: 0.845, `39`: 0.106, `26`: 0.030 |
| 1024 | `39` | `39`: 1.000 |

Modal apps: `ap-mbwACiwCaXAn4rsgjn3RCf` for 16-256 sims and
`ap-PrXen8rkfdz4Yno2Q2tBqQ` for 512-1024 sims.

Calibration read: the original 32-sim ResNet action `46` is not a stable
high-budget preference. At this state, both checkpoints' proposed moves change
as the simulation budget increases, and `action` can differ from the largest
`action_weights` entry because MCTX uses separate Gumbel/sequential-halving
selection for the played move. The 32-sim ResNet sweep should therefore be
treated as an evaluator artifact, not evidence that the ResNet checkpoint is
better than the transformer checkpoint.
