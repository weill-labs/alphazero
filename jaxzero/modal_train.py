"""Optional Modal app for JAX AlphaZero GPU training."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Mapping

from jaxzero.game_specs import (
    DEFAULT_GAME,
    resolve_game,
    resolve_network_defaults,
    supported_games,
)

WANDB_PROJECT_PREFIX = "alphazero"
_DEFAULT_GPU = "A10G"
_CHECKPOINT_VOLUME_NAME = "alphazero-checkpoints"
_CHECKPOINT_MOUNT = "/checkpoints"
_AUTO_MAX_STEPS = -1
_AUTO_SOLVER_EVAL_POSITIONS = -1

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
    try:
        return resolve_game(game).name
    except ValueError as exc:
        supported = ", ".join(supported_games())
        raise ValueError(
            f"jaxzero Modal training supports games: {supported}; got {game!r}"
        ) from exc


def _resolve_max_steps(game: str, max_steps: int) -> int:
    if max_steps == _AUTO_MAX_STEPS:
        return resolve_game(game).default_max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return max_steps


def _resolve_solver_eval_positions(game: str, solver_eval_positions: int) -> int:
    spec = resolve_game(game)
    if solver_eval_positions == _AUTO_SOLVER_EVAL_POSITIONS:
        return 64 if spec.supports_solver_eval else 0
    if solver_eval_positions < 0:
        raise ValueError("solver_eval_positions must be non-negative")
    if solver_eval_positions > 0 and not spec.supports_solver_eval:
        raise ValueError(
            "solver_eval_positions requires C4 solver support; "
            f"game {spec.name!r} has none"
        )
    return solver_eval_positions


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


def _validate_run_tag(run_tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_tag):
        raise ValueError("run_tag must contain only letters, numbers, '.', '_' or '-'")
    return run_tag


def _checkpoint_run_tag(wandb_run, seed: int, run_tag: str | None = None) -> str:
    if run_tag is not None:
        return _validate_run_tag(run_tag)
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


def _function_call_id(function_call) -> str:
    return str(
        getattr(
            function_call,
            "object_id",
            getattr(function_call, "id", function_call),
        )
    )


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
        # 12h: large-batch / high-sim runs (e.g. batch=512 sims=256 iters=150)
        # need ~6.25h, which the old 6h cap cut short mid-run. 12h is ~2x the
        # measured need so a slow run finishes, while still bounding a hang.
        timeout=12 * 60 * 60,
        secrets=[modal.Secret.from_name("wandb")],
        volumes={_CHECKPOINT_MOUNT: checkpoint_volume},
    )
    def train_remote(
        game: str = DEFAULT_GAME,
        iterations: int = 10,
        batch_size: int = 32,
        num_simulations: int = 32,
        max_steps: int = _AUTO_MAX_STEPS,
        selfplay_temperature: float = 1.0,
        selfplay_temperature_drop_step: int | None = None,
        selfplay_temperature_after_drop: float = 1.0,
        selfplay_dirichlet_fraction: float = 0.25,
        selfplay_dirichlet_fraction_drop_step: int | None = None,
        selfplay_dirichlet_fraction_after_drop: float = 0.25,
        selfplay_dirichlet_alpha: float = 0.3,
        channels: int = 64,
        num_res_blocks: int = 5,
        learning_rate: float = 1e-3,
        minibatch_size: int = 1024,
        init_checkpoint: str | None = None,
        checkpoint_every: int | None = None,
        eval_interval: int | None = None,
        eval_games: int = 64,
        replay_capacity: int | None = None,
        gating_interval: int | None = None,
        gating_games: int = 20,
        gating_threshold: float = 0.55,
        value_loss_weight: float = 1.0,
        mirror_augment: bool = False,
        solver_rehearsal_positions: int = 0,
        solver_rehearsal_batch_size: int = 0,
        solver_rehearsal_interval: int = 1,
        solver_rehearsal_seed: int | None = None,
        solver_rehearsal_target: str = "score",
        solver_rehearsal_solver_max_nodes: int = 250_000,
        solver_rehearsal_policy_loss_weight: float = 1.0,
        solver_rehearsal_value_loss_weight: float = 1.0,
        solver_rehearsal_hard_checkpoint: str | None = None,
        solver_rehearsal_hard_pool_size: int = 0,
        solver_rehearsal_hard_sims: int = 800,
        solver_rehearsal_anchor_positions: int = 0,
        weight_decay: float = 0.0,
        arch: str | None = None,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 512,
        use_value_cls_token: bool | None = None,
        policy_head_style: str | None = None,
        input_embed_style: str | None = None,
        solver_eval_positions: int = _AUTO_SOLVER_EVAL_POSITIONS,
        eval_sims: int = 64,
        seed: int = 0,
        requested_gpu: str = _DEFAULT_GPU,
        run_tag: str | None = None,
    ) -> dict[str, object]:
        from jaxzero.train import TrainingConfig, run_training

        game = _validate_game(game)
        if run_tag is not None:
            run_tag = _validate_run_tag(run_tag)
        max_steps = _resolve_max_steps(game, max_steps)
        solver_eval_positions = _resolve_solver_eval_positions(
            game, solver_eval_positions
        )
        network_defaults = resolve_network_defaults(
            game,
            arch=arch,
            use_value_cls_token=use_value_cls_token,
            policy_head_style=policy_head_style,
            input_embed_style=input_embed_style,
        )
        arch = str(network_defaults["arch"])
        use_value_cls_token = bool(network_defaults["use_value_cls_token"])
        policy_head_style = str(network_defaults["policy_head_style"])
        input_embed_style = str(network_defaults["input_embed_style"])
        run_config = {
            "game": game,
            "iterations": iterations,
            "batch_size": batch_size,
            "num_simulations": num_simulations,
            "max_steps": max_steps,
            "selfplay_temperature": selfplay_temperature,
            "selfplay_temperature_drop_step": selfplay_temperature_drop_step,
            "selfplay_temperature_after_drop": selfplay_temperature_after_drop,
            "selfplay_dirichlet_fraction": selfplay_dirichlet_fraction,
            "selfplay_dirichlet_fraction_drop_step": (
                selfplay_dirichlet_fraction_drop_step
            ),
            "selfplay_dirichlet_fraction_after_drop": (
                selfplay_dirichlet_fraction_after_drop
            ),
            "selfplay_dirichlet_alpha": selfplay_dirichlet_alpha,
            "channels": channels,
            "num_res_blocks": num_res_blocks,
            "learning_rate": learning_rate,
            "minibatch_size": minibatch_size,
            "init_checkpoint": init_checkpoint,
            "checkpoint_every": checkpoint_every,
            "eval_interval": eval_interval,
            "eval_games": eval_games,
            "replay_capacity": replay_capacity,
            "gating_interval": gating_interval,
            "gating_games": gating_games,
            "gating_threshold": gating_threshold,
            "value_loss_weight": value_loss_weight,
            "mirror_augment": mirror_augment,
            "solver_rehearsal_positions": solver_rehearsal_positions,
            "solver_rehearsal_batch_size": solver_rehearsal_batch_size,
            "solver_rehearsal_interval": solver_rehearsal_interval,
            "solver_rehearsal_seed": solver_rehearsal_seed,
            "solver_rehearsal_target": solver_rehearsal_target,
            "solver_rehearsal_solver_max_nodes": solver_rehearsal_solver_max_nodes,
            "solver_rehearsal_policy_loss_weight": (
                solver_rehearsal_policy_loss_weight
            ),
            "solver_rehearsal_value_loss_weight": solver_rehearsal_value_loss_weight,
            "solver_rehearsal_hard_checkpoint": solver_rehearsal_hard_checkpoint,
            "solver_rehearsal_hard_pool_size": solver_rehearsal_hard_pool_size,
            "solver_rehearsal_hard_sims": solver_rehearsal_hard_sims,
            "solver_rehearsal_anchor_positions": solver_rehearsal_anchor_positions,
            "weight_decay": weight_decay,
            "arch": arch,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_dim": mlp_dim,
            "use_value_cls_token": use_value_cls_token,
            "policy_head_style": policy_head_style,
            "input_embed_style": input_embed_style,
            "solver_eval_positions": solver_eval_positions,
            "eval_sims": eval_sims,
            "seed": seed,
            "requested_gpu": requested_gpu,
            "run_tag": run_tag,
        }
        run_name = (
            f"jaxzero-modal-{game}-{run_tag}"
            if run_tag is not None
            else f"jaxzero-modal-{game}-seed-{seed}"
        )
        wandb_run = _wandb_init(
            project=_wandb_project_for_game(game),
            run_name=run_name,
            config=run_config,
        )
        _print_wandb_url(wandb_run)
        checkpoint_run_tag = _checkpoint_run_tag(wandb_run, seed, run_tag=run_tag)
        checkpoint_path, checkpoint_dir = _resolve_checkpoint_paths(
            game, checkpoint_run_tag
        )
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
                    game=game,
                    iterations=iterations,
                    batch_size=batch_size,
                    num_simulations=num_simulations,
                    max_steps=max_steps,
                    selfplay_temperature=selfplay_temperature,
                    selfplay_temperature_drop_step=selfplay_temperature_drop_step,
                    selfplay_temperature_after_drop=selfplay_temperature_after_drop,
                    selfplay_dirichlet_fraction=selfplay_dirichlet_fraction,
                    selfplay_dirichlet_fraction_drop_step=(
                        selfplay_dirichlet_fraction_drop_step
                    ),
                    selfplay_dirichlet_fraction_after_drop=(
                        selfplay_dirichlet_fraction_after_drop
                    ),
                    selfplay_dirichlet_alpha=selfplay_dirichlet_alpha,
                    channels=channels,
                    num_res_blocks=num_res_blocks,
                    learning_rate=learning_rate,
                    minibatch_size=minibatch_size,
                    init_checkpoint=init_checkpoint,
                    checkpoint_every=checkpoint_every,
                    eval_interval=eval_interval,
                    eval_games=eval_games,
                    replay_capacity=replay_capacity,
                    gating_interval=gating_interval,
                    gating_games=gating_games,
                    gating_threshold=gating_threshold,
                    value_loss_weight=value_loss_weight,
                    mirror_augment=mirror_augment,
                    solver_rehearsal_positions=solver_rehearsal_positions,
                    solver_rehearsal_batch_size=solver_rehearsal_batch_size,
                    solver_rehearsal_interval=solver_rehearsal_interval,
                    solver_rehearsal_seed=solver_rehearsal_seed,
                    solver_rehearsal_target=solver_rehearsal_target,
                    solver_rehearsal_solver_max_nodes=solver_rehearsal_solver_max_nodes,
                    solver_rehearsal_policy_loss_weight=(
                        solver_rehearsal_policy_loss_weight
                    ),
                    solver_rehearsal_value_loss_weight=(
                        solver_rehearsal_value_loss_weight
                    ),
                    solver_rehearsal_hard_checkpoint=solver_rehearsal_hard_checkpoint,
                    solver_rehearsal_hard_pool_size=solver_rehearsal_hard_pool_size,
                    solver_rehearsal_hard_sims=solver_rehearsal_hard_sims,
                    solver_rehearsal_anchor_positions=(
                        solver_rehearsal_anchor_positions
                    ),
                    weight_decay=weight_decay,
                    arch=arch,
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    use_value_cls_token=use_value_cls_token,
                    policy_head_style=policy_head_style,
                    input_embed_style=input_embed_style,
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
        game: str = DEFAULT_GAME,
        iterations: int = 10,
        batch_size: int = 32,
        num_simulations: int = 32,
        max_steps: int = _AUTO_MAX_STEPS,
        selfplay_temperature: float = 1.0,
        selfplay_temperature_drop_step: int | None = None,
        selfplay_temperature_after_drop: float = 1.0,
        selfplay_dirichlet_fraction: float = 0.25,
        selfplay_dirichlet_fraction_drop_step: int | None = None,
        selfplay_dirichlet_fraction_after_drop: float = 0.25,
        selfplay_dirichlet_alpha: float = 0.3,
        channels: int = 64,
        num_res_blocks: int = 5,
        learning_rate: float = 1e-3,
        minibatch_size: int = 1024,
        init_checkpoint: str | None = None,
        checkpoint_every: int | None = None,
        eval_interval: int | None = None,
        eval_games: int = 64,
        replay_capacity: int | None = None,
        gating_interval: int | None = None,
        gating_games: int = 20,
        gating_threshold: float = 0.55,
        value_loss_weight: float = 1.0,
        mirror_augment: bool = False,
        solver_rehearsal_positions: int = 0,
        solver_rehearsal_batch_size: int = 0,
        solver_rehearsal_interval: int = 1,
        solver_rehearsal_seed: int | None = None,
        solver_rehearsal_target: str = "score",
        solver_rehearsal_solver_max_nodes: int = 250_000,
        solver_rehearsal_policy_loss_weight: float = 1.0,
        solver_rehearsal_value_loss_weight: float = 1.0,
        solver_rehearsal_hard_checkpoint: str | None = None,
        solver_rehearsal_hard_pool_size: int = 0,
        solver_rehearsal_hard_sims: int = 800,
        solver_rehearsal_anchor_positions: int = 0,
        weight_decay: float = 0.0,
        arch: str | None = None,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 512,
        use_value_cls_token: bool | None = None,
        policy_head_style: str | None = None,
        input_embed_style: str | None = None,
        solver_eval_positions: int = _AUTO_SOLVER_EVAL_POSITIONS,
        eval_sims: int = 64,
        seed: int = 0,
        gpu: str = _DEFAULT_GPU,
        run_tag: str | None = None,
        spawn: bool = False,
    ) -> None:
        remote_train = (
            train_remote.with_options(gpu=gpu) if gpu != _DEFAULT_GPU else train_remote
        )
        kwargs = dict(
            game=game,
            iterations=iterations,
            batch_size=batch_size,
            num_simulations=num_simulations,
            max_steps=max_steps,
            selfplay_temperature=selfplay_temperature,
            selfplay_temperature_drop_step=selfplay_temperature_drop_step,
            selfplay_temperature_after_drop=selfplay_temperature_after_drop,
            selfplay_dirichlet_fraction=selfplay_dirichlet_fraction,
            selfplay_dirichlet_fraction_drop_step=selfplay_dirichlet_fraction_drop_step,
            selfplay_dirichlet_fraction_after_drop=(
                selfplay_dirichlet_fraction_after_drop
            ),
            selfplay_dirichlet_alpha=selfplay_dirichlet_alpha,
            channels=channels,
            num_res_blocks=num_res_blocks,
            learning_rate=learning_rate,
            minibatch_size=minibatch_size,
            init_checkpoint=init_checkpoint,
            checkpoint_every=checkpoint_every,
            eval_interval=eval_interval,
            eval_games=eval_games,
            replay_capacity=replay_capacity,
            gating_interval=gating_interval,
            gating_games=gating_games,
            gating_threshold=gating_threshold,
            value_loss_weight=value_loss_weight,
            mirror_augment=mirror_augment,
            solver_rehearsal_positions=solver_rehearsal_positions,
            solver_rehearsal_batch_size=solver_rehearsal_batch_size,
            solver_rehearsal_interval=solver_rehearsal_interval,
            solver_rehearsal_seed=solver_rehearsal_seed,
            solver_rehearsal_target=solver_rehearsal_target,
            solver_rehearsal_solver_max_nodes=solver_rehearsal_solver_max_nodes,
            solver_rehearsal_policy_loss_weight=solver_rehearsal_policy_loss_weight,
            solver_rehearsal_value_loss_weight=solver_rehearsal_value_loss_weight,
            solver_rehearsal_hard_checkpoint=solver_rehearsal_hard_checkpoint,
            solver_rehearsal_hard_pool_size=solver_rehearsal_hard_pool_size,
            solver_rehearsal_hard_sims=solver_rehearsal_hard_sims,
            solver_rehearsal_anchor_positions=solver_rehearsal_anchor_positions,
            weight_decay=weight_decay,
            arch=arch,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            use_value_cls_token=use_value_cls_token,
            policy_head_style=policy_head_style,
            input_embed_style=input_embed_style,
            solver_eval_positions=solver_eval_positions,
            eval_sims=eval_sims,
            seed=seed,
            requested_gpu=gpu,
            run_tag=run_tag,
        )
        if spawn:
            function_call = remote_train.spawn(**kwargs)
            print(
                json.dumps(
                    {
                        "function_call_id": _function_call_id(function_call),
                        "game": game,
                        "run_tag": run_tag,
                        "gpu": gpu,
                    },
                    sort_keys=True,
                )
            )
            return

        result = remote_train.remote(**kwargs)
        print(json.dumps(result, indent=2, sort_keys=True))
