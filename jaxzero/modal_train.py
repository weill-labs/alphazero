"""Optional Modal app for JAX AlphaZero GPU training."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping

WANDB_PROJECT_PREFIX = "alphazero"
_DEFAULT_GPU = "A10G"
_CHECKPOINT_VOLUME_NAME = "alphazero-checkpoints"
_CHECKPOINT_MOUNT = "/checkpoints"
_SUPPORTED_GAME = "connectfour"

try:
    import modal
except ModuleNotFoundError:
    modal = None


def _modal_missing() -> RuntimeError:
    return RuntimeError(
        "Modal is optional. Install it with `uv sync --extra modal` before "
        "running `modal run jaxzero/modal_train.py`."
    )


def _validate_game(game: str) -> str:
    if game != _SUPPORTED_GAME:
        raise ValueError(
            f"jaxzero Modal training supports only {_SUPPORTED_GAME!r}; got {game!r}"
        )
    return game


def _wandb_project_for_game(game: str) -> str:
    return f"{WANDB_PROJECT_PREFIX}-{game}"


def _wandb_init(
    *,
    project: str,
    run_name: str,
    config: Mapping[str, object],
):
    try:
        import wandb

        return wandb.init(
            project=project,
            name=run_name,
            config=dict(config),
        )
    except Exception as exc:
        print(f"Warning: wandb disabled: {exc}", file=sys.stderr)
        return None


def _print_wandb_url(run) -> None:
    if run is None:
        return
    url = getattr(run, "url", None)
    if url:
        print(f"wandb run: {url}")


def _wandb_log(run, metrics: Mapping[str, int | float], *, step: int) -> None:
    if run is None:
        return
    try:
        run.log(dict(metrics), step=step)
    except Exception as exc:
        print(f"Warning: wandb log skipped: {exc}", file=sys.stderr)


def _wandb_finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"Warning: wandb finish skipped: {exc}", file=sys.stderr)


def _checkpoint_run_tag(wandb_run, seed: int) -> str:
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        return str(run_id)
    return f"seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}"


def _resolve_checkpoint_paths(
    game: str,
    run_tag: str,
    *,
    root: str = _CHECKPOINT_MOUNT,
) -> tuple[str, str]:
    """Return ``(final_checkpoint_path, checkpoint_dir)`` on the Modal Volume."""

    game = _validate_game(game)
    checkpoint_dir = f"{root}/{run_tag}/{game}"
    final_path = f"{checkpoint_dir}/final.msgpack"
    return final_path, checkpoint_dir


def _last_metrics(metrics: list[dict[str, float | int]]) -> dict[str, float | int]:
    return dict(metrics[-1]) if metrics else {}


if modal is None:
    app = None
    image = None
    checkpoint_volume = None

    def train_remote(*args, **kwargs):
        raise _modal_missing()

    def main(*args, **kwargs) -> None:
        raise _modal_missing()

else:
    app = modal.App("jaxzero")
    checkpoint_volume = modal.Volume.from_name(
        _CHECKPOINT_VOLUME_NAME, create_if_missing=True
    )
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "jax[cuda12]",
            "pgx>=2.6.0",
            "mctx>=0.0.6",
            "flax>=0.12.7",
            "optax>=0.2.8",
            "wandb>=0.27.0",
        )
        .add_local_python_source("jaxzero", "alphazero")
    )

    @app.function(
        image=image,
        gpu=_DEFAULT_GPU,
        timeout=6 * 60 * 60,
        secrets=[modal.Secret.from_name("wandb")],
        volumes={_CHECKPOINT_MOUNT: checkpoint_volume},
    )
    def train_remote(
        game: str = _SUPPORTED_GAME,
        iterations: int = 10,
        batch_size: int = 32,
        num_simulations: int = 32,
        max_steps: int = 64,
        channels: int = 64,
        num_res_blocks: int = 5,
        learning_rate: float = 1e-3,
        minibatch_size: int = 1024,
        init_checkpoint: str | None = None,
        checkpoint_every: int | None = None,
        eval_interval: int | None = None,
        eval_games: int = 64,
        replay_capacity: int | None = None,
        solver_eval_positions: int = 0,
        eval_sims: int = 64,
        seed: int = 0,
        requested_gpu: str = _DEFAULT_GPU,
    ) -> dict[str, object]:
        from jaxzero.train import TrainingConfig, run_training

        game = _validate_game(game)
        run_config = {
            "game": game,
            "iterations": iterations,
            "batch_size": batch_size,
            "num_simulations": num_simulations,
            "max_steps": max_steps,
            "channels": channels,
            "num_res_blocks": num_res_blocks,
            "learning_rate": learning_rate,
            "minibatch_size": minibatch_size,
            "init_checkpoint": init_checkpoint,
            "checkpoint_every": checkpoint_every,
            "eval_interval": eval_interval,
            "eval_games": eval_games,
            "replay_capacity": replay_capacity,
            "solver_eval_positions": solver_eval_positions,
            "eval_sims": eval_sims,
            "seed": seed,
            "requested_gpu": requested_gpu,
        }
        wandb_run = _wandb_init(
            project=_wandb_project_for_game(game),
            run_name=f"jaxzero-modal-{game}-seed-{seed}",
            config=run_config,
        )
        _print_wandb_url(wandb_run)
        run_tag = _checkpoint_run_tag(wandb_run, seed)
        checkpoint_path, checkpoint_dir = _resolve_checkpoint_paths(game, run_tag)
        print(f"checkpoint: {checkpoint_path} (volume {_CHECKPOINT_VOLUME_NAME})")
        try:
            training_started = time.perf_counter()

            def _log_iteration(metrics: dict[str, float | int]) -> None:
                # Stream each iteration's metrics live so the wandb charts
                # update during the run (not in one batch at the end).
                _wandb_log(wandb_run, metrics, step=int(metrics.get("iteration", 0)))

            def _commit_on_checkpoint(path: str) -> None:
                # Flush each periodic checkpoint to the Volume so it is durable
                # and downloadable mid-run (certifiable before the run finishes).
                checkpoint_volume.commit()

            extra_evaluator = None
            if solver_eval_positions > 0:
                from alphazero.c4_certify import make_solver_evaluator

                extra_evaluator = make_solver_evaluator(
                    sample_size=solver_eval_positions, sims=eval_sims, seed=seed
                )

            result = run_training(
                TrainingConfig(
                    iterations=iterations,
                    batch_size=batch_size,
                    num_simulations=num_simulations,
                    max_steps=max_steps,
                    channels=channels,
                    num_res_blocks=num_res_blocks,
                    learning_rate=learning_rate,
                    minibatch_size=minibatch_size,
                    init_checkpoint=init_checkpoint,
                    checkpoint_every=checkpoint_every,
                    eval_interval=eval_interval,
                    eval_games=eval_games,
                    replay_capacity=replay_capacity,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                ),
                on_iteration=_log_iteration,
                on_checkpoint=_commit_on_checkpoint,
                extra_evaluator=extra_evaluator,
            )
            training_seconds = max(time.perf_counter() - training_started, 1e-12)

            final_metrics = _last_metrics(result.metrics)
            modal_metrics = {
                "modal_training_seconds": training_seconds,
                "modal_iters_per_sec": iterations / training_seconds,
                "checkpoint_written": 1,
            }
            _wandb_log(wandb_run, modal_metrics, step=iterations)
            return {
                "metrics": result.metrics,
                "final_metrics": final_metrics,
                "modal_metrics": modal_metrics,
                "checkpoint_path": checkpoint_path,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_volume": _CHECKPOINT_VOLUME_NAME,
                "config": run_config,
            }
        finally:
            checkpoint_volume.commit()
            _wandb_finish(wandb_run)

    @app.local_entrypoint()
    def main(
        game: str = _SUPPORTED_GAME,
        iterations: int = 10,
        batch_size: int = 32,
        num_simulations: int = 32,
        max_steps: int = 64,
        channels: int = 64,
        num_res_blocks: int = 5,
        learning_rate: float = 1e-3,
        minibatch_size: int = 1024,
        init_checkpoint: str | None = None,
        checkpoint_every: int | None = None,
        eval_interval: int | None = None,
        eval_games: int = 64,
        replay_capacity: int | None = None,
        solver_eval_positions: int = 0,
        eval_sims: int = 64,
        seed: int = 0,
        gpu: str = _DEFAULT_GPU,
    ) -> None:
        remote_train = (
            train_remote.with_options(gpu=gpu) if gpu != _DEFAULT_GPU else train_remote
        )
        result = remote_train.remote(
            game=game,
            iterations=iterations,
            batch_size=batch_size,
            num_simulations=num_simulations,
            max_steps=max_steps,
            channels=channels,
            num_res_blocks=num_res_blocks,
            learning_rate=learning_rate,
            minibatch_size=minibatch_size,
            init_checkpoint=init_checkpoint,
            checkpoint_every=checkpoint_every,
            eval_interval=eval_interval,
            eval_games=eval_games,
            replay_capacity=replay_capacity,
            solver_eval_positions=solver_eval_positions,
            eval_sims=eval_sims,
            seed=seed,
            requested_gpu=gpu,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
