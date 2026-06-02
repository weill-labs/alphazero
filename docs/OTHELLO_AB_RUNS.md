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
