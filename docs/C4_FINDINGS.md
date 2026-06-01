# Connect Four AlphaZero: Findings Report

A complete account of the effort to drive the Connect Four blunder rate toward
"solved." This is the **results/conclusions** document; the companion
[C4_SOLVING.md](C4_SOLVING.md) is the **methods/how-to** (metric definitions,
the certification harness, CLI recipes). For the earlier seed-variance study see
[C4_ROBUSTNESS.md](C4_ROBUSTNESS.md).

## TL;DR

We drove the weak-blunder rate from a baseline **0.082 down to ~0.04** and then
hit a hard floor. Every algorithmic lever we tested — architecture, value-head
representation, eval search depth, self-play search depth, training length, and
batch size — moves *some* metric but **none break the ~0.04–0.07 floor**. A
deliberate SOTA-scale capstone run (batch=512, 150 iterations) confirmed the
floor holds at scale: **compute alone does not solve C4 in this setup.** The
floor sits ~10–20× above the literature's near-perfect agents (Prasad's weak err
≈ 0.24%), and the residual gap is a **setup/eval difference**, not a missing
knob.

The clearest scientific finding is a **decoupling**: several levers (the
transformer tower, deeper self-play search) sharply improve the value head's
calibration (`value_mae` from 0.94 down to 0.70) **without reducing blunders at
all**. At this plateau, blunders are governed by *policy fidelity on a small set
of sharp tactical positions*, and none of our knobs sharpen the policy there.

## The goal and the metric

"Solved" is defined as a near-zero, statistically-significant blunder rate
measured as **network + MCTS at eval `--sims 800`**, scored against the exact
Connect Four solver (`alphazero.c4_solver`). Search is part of the deployed
system, so we never measure the raw network policy — we measure the agent.

The headline metric is **`mean_wdl_regret`** (0–2 outcome-tier severity:
0 = result preserved, 1 = one tier lost like win→draw, 2 = catastrophic
win→loss). It is continuous and low-variance, so it discriminates checkpoints
that the binary weak `blunder_rate` (SE ≈ 0.03 on ~100 positions) cannot. All
numbers below are on a **fixed 256-position eval set** so comparisons are paired.
See [C4_SOLVING.md](C4_SOLVING.md) for the full metric rationale.

Reference points from the literature (different eval sets, so not directly
comparable, but order-of-magnitude anchors):

- **Prasad** (AlphaZero C4): weak err ≈ **0.24%**, strong err ≈ **3%**.
- **AlphaZero.jl**: evaluates at **1000 sims**, trains on **~5000 games/iter**.

## How we got to the floor (baseline progression)

Before the lever sweep, an earlier campaign (the `alphago-jla` goal, 29 Modal
A100 runs) drove the weak blunder rate from a **0.082 plateau to 0.036**
single-seed (z ≈ +2.16). The winning recipe — adopted as the baseline for
everything below — was:

- **128 channels** (vs 64): added capacity to break 0.08.
- **Pre-LayerNorm residual tower**: the unnormalized tower collapses at lr 5e-3;
  pre-LN fixes lr/depth stability.
- **`jla` = gating + best-net self-play**: only promote a new net if it beats the
  current best in a gating match; generate self-play data from the best net.
- **`--mirror-augment`**: 2× the policy training data via C4's horizontal
  symmetry. This was the single change that broke the 0.082 plateau, and only in
  composition with 128 channels.

A key earlier lesson that foreshadowed the final finding:
`--value-loss-weight 2.0` *alone* destabilizes the policy gradient and **hurt**
blunder rate; it only helps composed with mirror. And a run reaching the best
ever `value_mae` (0.847, better than baseline's 0.947) still showed an identical
0.082 blunder rate — **value calibration and blunder rate were already
decoupled.**

## The lever sweep — complete ledger

All entries are `mean_wdl_regret` / weak `blunder_rate` at `--sims 800` on the
fixed 256-position set. Architecture and WDLP are multi-seed; the
sims/iters/batch sweeps are single-seed but non-monotonic (which itself argues
no-effect). Bead IDs in the last column.

| Lever | Verdict | Evidence | Bead |
| --- | --- | --- | --- |
| Pre-LayerNorm tower | **adopted** | unnormalized tower collapses at lr 5e-3 | alphago-74e |
| 128ch vs 64ch | **helped break 0.08** | adopted in winning recipe; not the *floor* lever | alphago-q6e |
| jla + mirror augment | **adopted** | broke 0.082 → 0.036 single-seed | PR #46 |
| Naive transformer | **loses to ResNet** | wdl_regret 0.165 vs 0.098 | alphago-6xn |
| **Tier-1 transformer** (cls + per-column + conv3x3 + wd) | **matches ResNet** | 4-seed mean 0.098 = ResNet; t ≈ −4.1 vs naive | alphago-6xn, alphago-ixt |
| WDLP value head (W/D/L + ply) | **null** | paired Δ −0.004 (3 seeds); value_mae *worse* +0.016 | alphago-cut |
| Eval sims 64 → 800 | **saturates** | blunder barely moves; ~200 enough for C4's shallow tree | — |
| Self-play sims 32/64/128 | **null** | 0.072 / 0.108 / 0.073 wdl @ iters=80, non-monotonic | alphago-0fj |
| Training length 80 vs 200 iters | **converged by ~80** | 0.041 (iter80) ≈ 0.052 (iter200), same seed | alphago-0fj |
| Batch size 32/128/256 | **null** | 0.072 / 0.052 / 0.067 wdl @ iters=80, non-monotonic | alphago-2s9 |

Reading the table: the only changes that ever *helped* were the early
capacity/stability/data-augmentation choices that established the baseline. Once
on that baseline, **every knob we turned was null.**

## The transformer arc (full story across scales)

We invested heavily in a transformer tower because attention is SOTA on larger
boards. The arc:

1. **Naive transformer loses** (per-cell tokens + 2D position embeddings):
   wdl_regret 0.165 vs ResNet 0.098.
2. **Tier-1 board-aware redesign matches the ResNet**: adding a learnable
   value `[cls]` token (vs mean-pool), a column-shared policy head matching C4's
   action structure, a conv3×3 patch embedding before attention, and weight
   decay closed the entire gap — 4-seed mean wdl_regret 0.098, *equal* to the
   ResNet, t ≈ −4.1 vs the naive version. The only residual was a `value_mae`
   gap (0.812 vs ResNet 0.722).
3. **At SOTA scale, the ResNet pulls ahead again**: in the paired iter-150
   capstone (below), ResNet 0.072 wdl_regret **beats** transformer 0.113. The gap
   that was within-noise at small 4-seed scale (both 0.098) **widens with
   compute** — consistent with AlphaViT's finding that convolution beats
   attention on small boards (C4 is 7×6). The transformer kept only a `value_mae`
   edge (0.846 vs 0.935) that **did not convert to fewer blunders.**

Verdict: **the ResNet is the architecture for C4.** The transformer is a viable
match at small scale and an interesting candidate for larger boards
(Othello/Gomoku), but it is a dead end for *winning* on C4.

## The decisive capstone (SOTA-scale run)

The lever sweep ran on a ~30-minute experiment budget. The one experiment that
could change the headline was a single multi-hour SOTA-scale run — not another
knob, but **more total compute**. Calibrated cost on an A100 was ~72.7 s/iter at
batch=512/sims=256 (the batch dimension is nearly free because MCTS-across-games
is GPU-vectorized; sims dominate). A 150-iteration run is ~3 hr / ~$6–12.

We ran both arms paired at **batch=512, sims=256, 150 iterations, seed 500**
(~76k games, matching AlphaZero.jl's order of magnitude), certified at sims=800
on the fixed set (`alphago-qow`):

| Arm | weak blunder | mean_wdl_regret | policy_match | value_mae |
| --- | --- | --- | --- | --- |
| ResNet 128×5 | **0.046** | **0.072** | 95.4% | 0.935 |
| Transformer (Tier-1) | 0.077 | 0.113 | 92.3% | 0.846 |

**Two verdicts:**

1. **The floor held.** Neither arm broke below ~0.046 — the same band as the
   30-minute lever-sweep runs. *Compute scale alone does not solve C4 in this
   setup.* A properly-scaled run lands ~10–20× above the literature's
   near-perfect agents.
2. **ResNet beats transformer at scale** (see the transformer arc above).

This was the experiment that falsified the "the floor is just compute/data
volume" hypothesis that the earlier sweep had landed on.

## The residual-gap test (deeper self-play search)

One hypothesis for the residual gap: our self-play used few simulations
(producing weaker policy-improvement targets) vs AlphaZero.jl's ~600. We tested
it directly with a full-length **sims=600** ResNet run (batch=256, 100
iterations, seed 500), certified at sims=800 (`alphago-mgx`):

| | capstone sims=256 | **sims=600** | Δ |
| --- | --- | --- | --- |
| weak blunder | 0.046 | **0.041** (8/194) | tied |
| mean_wdl_regret | 0.072 | 0.067 | ~tied |
| policy_match | 95.4% | 95.9% | ~tied |
| **value_mae** | 0.935 | **0.700** | **−25%** |

The iter-90 and iter-100 certs were identical (converged by ~iter 90).

**Verdict:** deeper self-play search **dramatically improves value calibration**
(value_mae −25%) but **does not crack the blunder floor** — 0.041 vs 0.046 is one
position out of 194, statistically tied. The residual gap to the literature is
not explained by self-play sims either.

## The central scientific finding: value/policy decoupling

Three independent results point at the same conclusion:

- The early `value_mae` 0.847 run still blundered at 0.082.
- The transformer improved `value_mae` to 0.846 but blundered *more* (0.077).
- sims=600 improved `value_mae` to 0.700 (−25%) but blundered the same (0.041).

**At the C4 plateau, blunder rate is governed by policy fidelity on a small set
of sharp tactical positions — not by value-head calibration.** Levers that
sharpen the value head do not sharpen the policy on those positions, so they do
not reduce blunders. This is why "no cheap lever remains": we have no knob that
specifically improves policy accuracy on hard positions. The most promising
*untested* direction is therefore **solver-supervised hard-position rehearsal**
(open bead `alphago-fvh`) — directly training the policy on the positions it
gets wrong — rather than any of the global hyperparameters we swept.

## Operational lessons

- **Trust the offline ≥128-position cert, not the inline metric.** The inline
  `eval/c4_blunder_rate` is computed at a fixed seed on ~49 sampled positions; a
  net can consistently fail the same 1–2 of them and flat-line at a spurious
  value. One model reading 0.041 inline certified at 0.124 on 128 positions.
- **Replicate with ≥3 seeds below ~0.10 blunder.** Seed variance dominates there:
  the same config across seeds gave 0.042 / 0.073 / 0.177 — a 4× spread. Treat
  single-seed wins below 0.10 as suggestive only.
- **Fixed eval set + paired comparison** is what made the sweep tractable —
  far more statistical power than unpaired sampling, and the prerequisite for the
  continuous-regret metrics to be meaningful.
- **Parallel certification is routine now.** `--workers N` gives ~25× on a
  32-core host (byte-identical to serial); the `--batched` + cached-solver-labels
  path wins on GPU. Either makes the sims=800 verdict cheap.
- **Size spot-GPU runs to the eviction window, or make them resumable.** Modal
  A100 spot instances evict at ~6 hr. A batch=512/sims=600 run is ~330 s/iter
  (~9 hr for 100 iters) and **cannot finish** — we lost two runs at iter ~64–68.
  Dropping to batch=256 (~165 s/iter, ~4.6 hr; batch is a proven-null lever)
  completed cleanly. Don't fight a config that structurally can't survive the
  eviction window.

## Conclusion and what we'd do next

The C4 weak-blunder floor is **~0.04–0.07 at sims=800, robust to every
algorithmic lever and to compute scale.** It is structural to this setup, ~10–20×
above the literature's near-perfect agents. The residual gap is most likely a
**setup/eval difference** — eval-set difficulty, or a training-detail mismatch
vs AlphaZero.jl — not a missing global hyperparameter, and not raw compute.

Given the decoupling finding, global hyperparameter sweeps are exhausted. The
remaining open directions, in priority order:

1. **Solver-supervised hard-position rehearsal** (`alphago-fvh`) — train the
   policy directly on the tactical positions it blunders. This targets the actual
   bottleneck (policy fidelity), unlike everything swept so far.
2. **Eval-set / methodology reconciliation** — confirm whether the gap to the
   literature is real or an eval-set artifact (the open eval-set + cached-labels
   work in `alphago-yom` makes this a fixed, shareable comparison).
3. **Bigger boards** (Othello/Gomoku, `alphago-hjx`) — where AlphaViT suggests
   attention may finally beat convolution; requires Elo-based eval since the
   exact solver is C4-specific.

Do **not** re-run single-lever sweeps (arch, value-head, sims, iters, batch) or
the transformer on C4 — all are settled.

## Reproduction

The winning C4 recipe (ResNet) and the exact CLI flags for training and
certification are in [C4_SOLVING.md](C4_SOLVING.md). The fixed eval set and
cached solver labels make every cert a paired comparison — build them once with
`--build-eval-set` / `--build-eval-labels` and reuse across all runs.

## Bead trail

- `alphago-1aq` — epic: Solve Connect Four
- `alphago-6xn`, `alphago-ixt` — transformer A/B (naive loses; Tier-1 matches)
- `alphago-cut` — WDLP value head (null)
- `alphago-0fj` — self-play sims sweep + training-length (null)
- `alphago-2s9` — batch-size ladder (null)
- `alphago-qow` — SOTA-scale capstone (floor held; ResNet > transformer)
- `alphago-mgx` — sims=600 residual-gap test (value_mae −25%, floor held)
- `alphago-fvh` — **open**: solver-supervised hard-position rehearsal (next lever)
