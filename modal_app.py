"""Optional Modal app for cloud tic-tac-toe AlphaZero training."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

WANDB_PROJECT = "alphazero-tictactoe"

try:
    import modal
except ModuleNotFoundError:
    modal = None


def _modal_missing() -> RuntimeError:
    return RuntimeError(
        "Modal is optional. Install it with `uv sync --extra modal` before "
        "running `modal run modal_app.py`."
    )


def _wandb_init(
    *,
    run_name: str,
    config: Mapping[str, int | str | None],
):
    try:
        import wandb

        return wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=dict(config),
        )
    except Exception as exc:
        print(f"Warning: wandb disabled: {exc}", file=sys.stderr)
        return None


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


if modal is None:
    app = None
    image = None

    def train_remote(*args, **kwargs):
        raise _modal_missing()

    def main(*args, **kwargs) -> None:
        raise _modal_missing()

else:
    app = modal.App("alphazero-tictactoe")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch>=2.2", "numpy>=1.26", "wandb>=0.27.0")
        .add_local_python_source("alphazero")
    )

    @app.function(
        image=image,
        timeout=6 * 60 * 60,
        secrets=[modal.Secret.from_name("wandb")],
    )
    def train_remote(
        iterations: int = 35,
        self_play_games: int = 8,
        sims: int = 96,
        seed: int = 0,
        gpu: str | None = None,
        eval_games: int = 40,
        eval_sims: int = 200,
    ) -> dict[str, object]:
        from alphazero.arena import (
            MCTSPlayer,
            PerfectPlayer,
            RandomPlayer,
            play_match,
            train_tictactoe_agent,
        )
        from alphazero.games.tictactoe import TicTacToe

        run_config = {
            "iterations": iterations,
            "self_play_games": self_play_games,
            "self_play_sims": sims,
            "seed": seed,
            "requested_gpu": gpu,
            "eval_games": eval_games,
            "eval_sims": eval_sims,
        }
        wandb_run = _wandb_init(
            run_name=f"modal-tictactoe-seed-{seed}",
            config=run_config,
        )
        try:
            net, metrics = train_tictactoe_agent(
                iterations=iterations,
                self_play_games_per_iteration=self_play_games,
                self_play_mcts_cfg={
                    "num_simulations": sims,
                    "dirichlet_eps": 0.25,
                },
                checkpoint_path=None,
                seed=seed,
                wandb_run=wandb_run,
                wandb_config=run_config,
            )
            game = TicTacToe()
            perfect_wins, perfect_draws, perfect_losses = play_match(
                MCTSPlayer(net, num_simulations=eval_sims, seed=seed),
                PerfectPlayer(),
                game,
                eval_games,
            )
            random_wins, random_draws, random_losses = play_match(
                MCTSPlayer(net, num_simulations=eval_sims, seed=seed + 1),
                RandomPlayer(seed=seed),
                game,
                eval_games,
            )
            eval_metrics = {
                "eval/perfect_wins": perfect_wins,
                "eval/perfect_draws": perfect_draws,
                "eval/perfect_losses": perfect_losses,
                "eval/random_wins": random_wins,
                "eval/random_draws": random_draws,
                "eval/random_losses": random_losses,
            }
            _wandb_log(wandb_run, eval_metrics, step=iterations)
            return {
                "metrics": metrics,
                "vs_perfect": {
                    "wins": perfect_wins,
                    "draws": perfect_draws,
                    "losses": perfect_losses,
                },
                "vs_random": {
                    "wins": random_wins,
                    "draws": random_draws,
                    "losses": random_losses,
                },
                "config": run_config,
            }
        finally:
            _wandb_finish(wandb_run)

    @app.local_entrypoint()
    def main(
        iterations: int = 35,
        self_play_games: int = 8,
        sims: int = 96,
        seed: int = 0,
        gpu: str | None = None,
        eval_games: int = 40,
        eval_sims: int = 200,
    ) -> None:
        remote_train = train_remote.with_options(gpu=gpu) if gpu else train_remote
        result = remote_train.remote(
            iterations=iterations,
            self_play_games=self_play_games,
            sims=sims,
            seed=seed,
            gpu=gpu,
            eval_games=eval_games,
            eval_sims=eval_sims,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
