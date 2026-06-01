# Solving Connect Four

How we measure and push Connect Four play quality, the network options, and a
decision log of what moved the blunder rate and what did not. Companion to the
older seed study in [C4_ROBUSTNESS.md](C4_ROBUSTNESS.md).

## Goal and definition of "solved"

Drive the **blunder rate** toward zero, measured against the exact Connect Four
solver (`alphazero.c4_solver`). "Solved" is defined as a near-zero
statistically-significant blunder rate measured as **net + MCTS at eval
`--sims 800`** — the deployment-realistic metric matching the literature
(AlphaZero.jl evals at 1000 sims; Prasad reports ~3% at high sims). Search is
part of the system; we do not measure the raw network policy.

## Certification harness and metrics (`alphazero/c4_certify.py`)

Each position is scored by comparing the agent's MCTS move (and value) to exact
solver labels. The metrics, in increasing order of usefulness for comparing
runs:

- **`blunder_rate`** (weak / outcome mode): fraction of positions whose
  game-theoretic W/D/L outcome the move changed. Matches Prasad's "weak" mode
  (~0.24% err for a strong agent). Saturates near zero, so it is a poor
  run-vs-run discriminator.
- **`score_blunder_rate`** (strong mode): fraction of moves that were not
  score-optimal, including winning slower than the fastest forced win. Always
  `>= blunder_rate`. Matches Prasad's "strong" mode (~3% err).
- **`mean_wdl_regret`** (0–2): outcome-tier severity — 0 preserves the result,
  1 downgrades one tier (win→draw or draw→loss), 2 is catastrophic (win→loss).
  Continuous and low-variance — the **preferred headline comparison metric**.
- **`mean_score_regret`**: raw Pons-score distance to optimal (penalizes slower
  wins / faster losses within a tier). Secondary signal.
- **`value_mae`**: value-head calibration error vs solver labels.

`solve_with_score()` exposes the solver's distance-to-win score so the
weak/strong split and both regrets are real (plain `solve()` collapses to
ternary W/D/L).

Certification defaults to deterministic perfect-information search:
`--gumbel-scale 0.0`. Older certs used Gumbel root noise (`--gumbel-scale 1.0`);
use that flag only when reproducing historical stochastic numbers.

### Why these metrics

Bare blunder rate on ~100 sampled positions has SE ≈ 0.03 (±3 positions at
p≈0.1) and is binary (a thrown-away forced win and a 2nd-best move in a dead
draw both score "1 blunder"). It cannot distinguish good checkpoints. The fix:

- **Fixed eval set** (`--build-eval-set` / `--eval-set`): every run is scored on
  the *same* positions, making comparisons paired (far more powerful than
  unpaired sampling).
- **Continuous, severity-aware regret**: lower variance per position, so smaller
  true differences are detectable and fewer seeds are needed.

### Performance

- **`--workers N`**: position-level parallel certification across a spawned
  process pool. ~25× on a 32-core host; results byte-identical to serial. Makes
  the `--sims 800` verdict routine.
- **`--batched` + cached solver labels** (`--build-eval-labels` / `--eval-labels`):
  the fixed eval set's solver labels never change, so precompute them once
  (incl. per-legal-child scores for regret) and run MCTS for all positions in a
  single `vmap`'d `mctx` call. ~10× per-cert vs single-process serial after a
  one-time label precompute; batched MCTS is 3.9× faster than serial batch-1 on
  CPU and dramatically faster on GPU. On a CPU box the `--workers` pool is
  comparable; the batched path wins on GPU and the label cache helps every path.

## Network architectures (`jaxzero/net.py`)

Both towers satisfy the same `(policy_logits, value)` contract, so MCTS,
self-play, and the cert are architecture-agnostic.

- **ResNet** (`--arch resnet`, default): pre-LN residual tower,
  `--channels` × `--num-res-blocks`. The 128ch × 5 net is the strong baseline.
- **Transformer** (`--arch transformer`): per-cell tokenization, learned 2D
  (row+col) position embeddings, pre-LN MHA+MLP blocks. The naive version loses
  to the ResNet; the **Tier-1 board-aware design matches it**:
  - `--use-value-cls-token`: BERT/ViT-style value token (vs mean-pool).
  - `--policy-head-style per_column`: column-shared head matching C4's action
    structure (requires `action_size == width`).
  - `--input-embed-style conv3x3`: local patch embedding before attention.

## Decision log

Blunder rates are `mean_wdl_regret` / weak `blunder_rate` at `--sims 800` on the
fixed 256-position set unless noted. Most single-lever verdicts are 1–3 seeds;
architecture and WDLP are multi-seed.

| Lever | Verdict | Evidence |
| --- | --- | --- |
| LayerNorm in residual tower | **adopted** | unnormalized tower collapses at lr 5e-3; pre-LN fixes lr/depth stability (alphago-74e) |
| 128ch vs 64ch capacity | **helped break 0.08** | 128ch adopted in the winning recipe; not the *floor* lever (alphago-q6e) |
| jla (gating + best-net self-play) + mirror augment | **adopted** | broke the 0.082 plateau to 0.036 single-seed (PR #46) |
| Naive transformer | **loses to ResNet** | 0.165 vs 0.098 `wdl_regret` |
| **Tier-1 transformer** (cls + per-column + conv3x3 + wd) | **matches ResNet** | 4-seed mean 0.098 = ResNet, t≈−4.1 vs naive (PR #47, alphago-6xn) |
| WDLP value head (W/D/L + ply) | **null** | paired Δ −0.004; `value_mae` no better. C4 outcomes already ternary (reverted) |
| Eval sims 64 → 800 | **saturates** | blunder barely moves; 200 already enough for C4's shallow tree |
| Self-play sims 32/64/128 | **null** | non-monotonic 0.041/0.067/0.041 at iters=80 — sampling noise |
| Training length 80 vs 200 iters | **converged by ~80** | iters=80 ≈ iters=200 for the same seed |
| Batch size 32/128/256 | **null** | non-monotonic 0.041/0.031/0.041 at iters=80 — sampling noise |
| Deterministic checkpoint ladder | **adopted** | capstone ResNet `iter_0050` beats final: 20/856 = 0.023 on 1024-sample cert |

## The blunder floor

The C4 weak-blunder floor for **final checkpoints** was **~0.04–0.07** at
`--sims 800`, and every global training lever above failed to move it
reliably. A later deterministic checkpoint ladder found a free checkpoint
selection win: the capstone ResNet `trjd57fm/iter_0050` certified at
**20/856 = 0.023** on a larger 1024-sample cert, better than both the same run's
final checkpoint (**29/857 = 0.034**) and the sims=600 final (**30/859 =
0.035**). This does not mean compute scale solved C4; it means periodic
checkpoint selection matters, and final checkpoint metrics can hide the best
model. The remaining bottleneck is still **policy fidelity on sharp tactical
positions**, not value calibration. See the full **[C4_FINDINGS.md](C4_FINDINGS.md)**
report for the complete results, the capstone, the sims=600 residual-gap test,
the checkpoint ladder, and the value/policy decoupling finding.

## SOTA-scale run: cost

Calibrated on an A100 at `--batch-size 512 --num-simulations 256`:
**~72.7 s/iter** steady-state. The batch dimension is nearly free (16× more
games for ~1.5× time — MCTS-across-games is GPU-vectorized); sims dominates
(sequential per move).

- `--batch-size 512 --num-simulations 256 --iterations 150`: ~3.0 hr → ~$6–12.
- `--num-simulations 128` (still 4× our baseline search): ~1.5 hr → ~$3–6.
- 3-seed confirmation: ~$10–35.

The blocker was never money — it is wall-clock-per-run (a single multi-hour
run), not the 30-min experiment budget used for the lever sweeps.

## Using the tools

```bash
# Build a fixed eval set once (reused for paired comparison across runs).
python -m alphazero.c4_certify --build-eval-set evalset.json --sample-size 256 --seed 0

# Precompute + cache solver labels for that set once (removes the solver
# bottleneck from every later cert).
python -m alphazero.c4_certify --build-eval-labels evalset.labels.json --eval-set evalset.json

# Certify a checkpoint at the solve-target sims, parallel across cores.
python -m alphazero.c4_certify --checkpoint final.msgpack --eval-set evalset.json \
  --sims 800 --workers 28

# Or the batched/cached path (fast per-cert; best on GPU).
python -m alphazero.c4_certify --checkpoint final.msgpack --eval-set evalset.json \
  --sims 800 --batched --eval-labels evalset.labels.json

# Certify every periodic checkpoint in a local directory and pick the best
# solver-scored checkpoint (mean_wdl_regret, then weak blunder).
python -m alphazero.c4_certify --checkpoint-dir checkpoints/run/connectfour \
  --eval-labels evalset.labels.json --batched --sims 800
```

Train a Tier-1 transformer on Modal:

```bash
modal run jaxzero/modal_train.py::main \
  --arch transformer --d-model 128 --num-layers 6 --num-heads 4 --mlp-dim 512 \
  --use-value-cls-token --policy-head-style per_column --input-embed-style conv3x3 \
  --weight-decay 1e-4 --mirror-augment \
  --gating-interval 10 --gating-games 20 --gating-threshold 0.55 \
  --iterations 200 --batch-size 32 --num-simulations 32 --seed 0 --gpu A100
```

## Future work

The SOTA-scale capstone is **done** (it held the floor — see
[C4_FINDINGS.md](C4_FINDINGS.md)). Global hyperparameter sweeps are exhausted.
The remaining directions, in priority order:

- **Solver-supervised hard-position rehearsal** (`alphago-fvh`) — train the
  policy directly on the tactical positions it blunders. Targets the actual
  bottleneck (policy fidelity), unlike every lever swept so far.
- **Eval-set / methodology reconciliation** — confirm whether the ~10–20× gap to
  the literature is real or an eval-set artifact, using a fixed shareable eval
  set + cached labels (`alphago-yom`).
- **Bigger boards** (Othello / Gomoku, `alphago-hjx`) — where AlphaViT suggests
  attention may beat convolution; requires non-solver (Elo-based) evaluation
  since the exact solver is C4-specific.
- **Inline batched eval** — wire the batched cert into the per-iteration training
  eval; marginal (small sample, infrequent) but removes the inline solver cost on
  the training container.
