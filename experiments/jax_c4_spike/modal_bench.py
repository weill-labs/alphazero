"""Run the JAX Connect Four self-play throughput spike on a Modal GPU.

bench.py established CPU parity (JAX ~= the PyTorch process pool on 8 cores).
JAX's batch-vectorization advantage is latent in the batch dimension and only
realizes on a GPU, so this runs the *identical* ``run_benchmark`` workload on a
Modal GPU with CUDA jax.

It sweeps several batch sizes in one GPU allocation: a small batch (apples to
apples with the CPU test) up to a large batch (the GPU throughput ceiling).

    cd experiments/jax_c4_spike
    uv run modal run modal_bench.py                              # A10G sweep
    uv run modal run modal_bench.py --gpu A100-40GB --batch-sizes 256,1024,4096
"""

from __future__ import annotations

import json

import modal

_DEFAULT_GPU = "A10G"
app = modal.App("jax-c4-gpu-bench")
# CUDA jax (the cuda12 pip wheels bundle the CUDA runtime, so debian_slim + a
# Modal GPU is enough). bench.py + network.py are shipped as local sources so
# the container runs exactly the code benchmarked on CPU.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "pgx>=2.0.0",
        "dm-haiku",
        "mctx",
        "optax",
    )
    .add_local_python_source("bench", "network")
)


@app.function(image=image, gpu=_DEFAULT_GPU, timeout=30 * 60)
def gpu_benchmark(
    batch_sizes: list[int],
    num_simulations: int,
    num_channels: int,
    num_blocks: int,
    max_steps: int,
    timed_iters: int,
) -> list[dict[str, object]]:
    from bench import run_benchmark

    results: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        try:
            results.append(
                run_benchmark(
                    batch_size=batch_size,
                    num_simulations=num_simulations,
                    num_channels=num_channels,
                    num_blocks=num_blocks,
                    max_steps=max_steps,
                    timed_iters=timed_iters,
                )
            )
        except Exception as exc:  # e.g. OOM at the largest batch
            # Record and continue so the smaller batches' results survive.
            results.append(
                {"batch_size": batch_size, "error": f"{type(exc).__name__}: {exc}"}
            )
    return results


@app.local_entrypoint()
def main(
    batch_sizes: str = "128,512,2048",
    num_simulations: int = 128,
    num_channels: int = 64,
    num_blocks: int = 5,
    max_steps: int = 64,
    timed_iters: int = 3,
    gpu: str = _DEFAULT_GPU,
) -> None:
    sizes = [int(part) for part in batch_sizes.split(",") if part.strip()]
    runner = (
        gpu_benchmark.with_options(gpu=gpu) if gpu != _DEFAULT_GPU else gpu_benchmark
    )
    results = runner.remote(
        batch_sizes=sizes,
        num_simulations=num_simulations,
        num_channels=num_channels,
        num_blocks=num_blocks,
        max_steps=max_steps,
        timed_iters=timed_iters,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
