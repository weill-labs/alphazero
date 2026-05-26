# Benchmarking AlphaZero

Use `alphazero.benchmark` to run a fixed self-play plus training workload and
measure where wall-clock time is going.

```bash
uv run python -m alphazero.benchmark --game tictactoe
uv run python -m alphazero.benchmark --game connectfour --self-play-games 2 --mcts-sims 32
```

Useful size flags:

```bash
uv run python -m alphazero.benchmark \
  --game tictactoe \
  --self-play-games 4 \
  --mcts-sims 128 \
  --train-epochs 4 \
  --batch-size 64 \
  --seed 0
```

The report includes:

- self-play/MCTS wall time
- network inference time inside MCTS
- non-inference MCTS overhead
- train-step time
- self-play games/sec
- MCTS net-evals/sec
- train steps/sec
- the dominant measured cost

Profiling mode wraps the same fixed workload in `cProfile` and prints the top
functions by cumulative time:

```bash
uv run python -m alphazero.benchmark --game tictactoe --profile --profile-top 30
```

The expected bottleneck for the current implementation is sequential
single-position network inference from MCTS leaf expansion. Each MCTS expansion
calls `net.predict` for one board at a time, so small matrix/convolution work
pays Python and PyTorch dispatch overhead repeatedly.

Candidate speedups to try:

- Batched or leaf-parallel MCTS: collect multiple leaf states, evaluate them in
  one network batch, then back up results.
- Vectorized self-play: run multiple games concurrently so MCTS leaf batches
  have enough work to amortize framework overhead.
- Use `torch.inference_mode()` and `eval()` for inference-only calls, then verify
  that training mode is restored exactly as before.
- Tune PyTorch intra-op threads with `--torch-threads`; small boards may be
  faster with fewer threads.
- GPU batching: only move inference to GPU once MCTS can provide meaningful
  batches. Single-position tic-tac-toe inference is usually too small to benefit.

This benchmark is eval-only infrastructure. It should not change training,
self-play, or evaluation results when timing hooks are not supplied.
