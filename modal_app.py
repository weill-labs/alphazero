"""Optional Modal app for cloud AlphaZero training."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence

WANDB_PROJECT = "alphazero-tictactoe"
_DEFAULT_GATING_INTERVAL = 5
_DEFAULT_GATING_GAMES = 20
_DEFAULT_GATING_THRESHOLD = 0.55
_DEFAULT_EVAL_INTERVAL = 5
_DEFAULT_LADDER_GAMES = 20
_DEFAULT_LADDER_DEPTHS = (1, 2, 4)
_DEFAULT_LADDER_DEPTHS_CLI = ",".join(str(depth) for depth in _DEFAULT_LADDER_DEPTHS)
_TICTACTOE_DEFAULTS = {
    "iterations": 60,
    "self_play_games": 24,
    "sims": 128,
}
_CONNECTFOUR_DEFAULTS = {
    "iterations": 120,
    "self_play_games": 48,
    "sims": 256,
}

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
    config: Mapping[str, object],
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


def _defaults_for_game(game: str) -> Mapping[str, int]:
    if game == "tictactoe":
        return _TICTACTOE_DEFAULTS
    if game == "connectfour":
        return _CONNECTFOUR_DEFAULTS
    raise ValueError("game must be 'tictactoe' or 'connectfour'")


def _resolve_training_args(
    *,
    game: str,
    iterations: int | None,
    self_play_games: int | None,
    sims: int | None,
) -> tuple[int, int, int]:
    defaults = _defaults_for_game(game)
    return (
        defaults["iterations"] if iterations is None else iterations,
        defaults["self_play_games"] if self_play_games is None else self_play_games,
        defaults["sims"] if sims is None else sims,
    )


def _parse_ladder_depths(ladder_depths: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(ladder_depths, str):
        depths = tuple(
            int(part.strip()) for part in ladder_depths.split(",") if part.strip()
        )
    else:
        depths = tuple(int(depth) for depth in ladder_depths)
    if not depths:
        raise ValueError("ladder_depths must contain at least one depth")
    if any(depth < 1 for depth in depths):
        raise ValueError("ladder_depths must all be at least 1")
    return depths


def _resolve_eval_args(
    *,
    gating_interval: int,
    gating_games: int,
    gating_threshold: float,
    eval_interval: int,
    ladder_games: int,
    ladder_depths: str | Sequence[int],
) -> dict[str, object]:
    return {
        "gating_interval": gating_interval,
        "gating_games": gating_games,
        "gating_threshold": gating_threshold,
        "eval_interval": eval_interval,
        "ladder_games": ladder_games,
        "ladder_depths": _parse_ladder_depths(ladder_depths),
    }


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
        game: str = "tictactoe",
        iterations: int | None = None,
        self_play_games: int | None = None,
        sims: int | None = None,
        seed: int = 0,
        gpu: str | None = None,
        eval_games: int = 40,
        eval_sims: int = 200,
        gating_interval: int = _DEFAULT_GATING_INTERVAL,
        gating_games: int = _DEFAULT_GATING_GAMES,
        gating_threshold: float = _DEFAULT_GATING_THRESHOLD,
        eval_interval: int = _DEFAULT_EVAL_INTERVAL,
        ladder_games: int = _DEFAULT_LADDER_GAMES,
        ladder_depths: str = _DEFAULT_LADDER_DEPTHS_CLI,
    ) -> dict[str, object]:
        from alphazero.arena import (
            MCTSPlayer,
            PerfectPlayer,
            RandomPlayer,
            evaluate_connect_four_tactics,
            play_match,
            train_agent,
            train_tictactoe_agent,
        )
        from alphazero.games.connectfour import ConnectFour
        from alphazero.games.tictactoe import TicTacToe

        iterations, self_play_games, sims = _resolve_training_args(
            game=game,
            iterations=iterations,
            self_play_games=self_play_games,
            sims=sims,
        )
        selected_game = ConnectFour() if game == "connectfour" else TicTacToe()
        eval_args = _resolve_eval_args(
            gating_interval=gating_interval,
            gating_games=gating_games,
            gating_threshold=gating_threshold,
            eval_interval=eval_interval,
            ladder_games=ladder_games,
            ladder_depths=ladder_depths,
        )
        run_config = {
            "game": game,
            "iterations": iterations,
            "self_play_games": self_play_games,
            "self_play_sims": sims,
            "seed": seed,
            "requested_gpu": gpu,
            "eval_games": eval_games,
            "eval_sims": eval_sims,
            **eval_args,
        }
        wandb_run = _wandb_init(
            run_name=f"modal-{game}-seed-{seed}",
            config=run_config,
        )
        _print_wandb_url(wandb_run)
        try:
            training_started = time.perf_counter()
            training_kwargs = {
                "iterations": iterations,
                "self_play_games_per_iteration": self_play_games,
                "self_play_mcts_cfg": {
                    "num_simulations": sims,
                    "dirichlet_eps": 0.25,
                },
                "checkpoint_path": None,
                "seed": seed,
                "wandb_run": wandb_run,
                "wandb_config": run_config,
                **eval_args,
            }
            if game == "tictactoe":
                net, metrics = train_tictactoe_agent(**training_kwargs)
            else:
                net, metrics = train_agent(selected_game, **training_kwargs)
            training_seconds = max(time.perf_counter() - training_started, 1e-12)
            metrics["modal_training_seconds"] = training_seconds
            metrics["modal_iters_per_sec"] = iterations / training_seconds
            metrics["modal_self_play_games_per_sec"] = (
                iterations * self_play_games / training_seconds
            )

            if game == "connectfour":
                agent = MCTSPlayer(net, num_simulations=eval_sims, seed=seed)
                tactical_metrics = evaluate_connect_four_tactics(agent, selected_game)
                random_wins, random_draws, random_losses = play_match(
                    agent,
                    RandomPlayer(seed=seed),
                    selected_game,
                    eval_games,
                )
                random_win_rate = random_wins / eval_games
                eval_metrics = {
                    "eval/c4_immediate_win_rate": tactical_metrics[
                        "immediate_win_rate"
                    ],
                    "eval/c4_block_rate": tactical_metrics["block_rate"],
                    "eval/c4_random_wins": random_wins,
                    "eval/c4_random_draws": random_draws,
                    "eval/c4_random_losses": random_losses,
                    "eval/c4_random_win_rate": random_win_rate,
                    "modal_training_seconds": metrics["modal_training_seconds"],
                    "modal_iters_per_sec": metrics["modal_iters_per_sec"],
                    "modal_self_play_games_per_sec": metrics[
                        "modal_self_play_games_per_sec"
                    ],
                }
                _wandb_log(wandb_run, eval_metrics, step=iterations)
                return {
                    "metrics": metrics,
                    "c4_tactics": tactical_metrics,
                    "vs_random": {
                        "wins": random_wins,
                        "draws": random_draws,
                        "losses": random_losses,
                        "win_rate": random_win_rate,
                    },
                    "config": run_config,
                }

            perfect_wins, perfect_draws, perfect_losses = play_match(
                MCTSPlayer(net, num_simulations=eval_sims, seed=seed),
                PerfectPlayer(),
                selected_game,
                eval_games,
            )
            random_wins, random_draws, random_losses = play_match(
                MCTSPlayer(net, num_simulations=eval_sims, seed=seed + 1),
                RandomPlayer(seed=seed),
                selected_game,
                eval_games,
            )
            eval_metrics = {
                "eval/perfect_wins": perfect_wins,
                "eval/perfect_draws": perfect_draws,
                "eval/perfect_losses": perfect_losses,
                "eval/random_wins": random_wins,
                "eval/random_draws": random_draws,
                "eval/random_losses": random_losses,
                "modal_training_seconds": metrics["modal_training_seconds"],
                "modal_iters_per_sec": metrics["modal_iters_per_sec"],
                "modal_self_play_games_per_sec": metrics[
                    "modal_self_play_games_per_sec"
                ],
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
        game: str = "tictactoe",
        iterations: int | None = None,
        self_play_games: int | None = None,
        sims: int | None = None,
        seed: int = 0,
        gpu: str | None = None,
        eval_games: int = 40,
        eval_sims: int = 200,
        gating_interval: int = _DEFAULT_GATING_INTERVAL,
        gating_games: int = _DEFAULT_GATING_GAMES,
        gating_threshold: float = _DEFAULT_GATING_THRESHOLD,
        eval_interval: int = _DEFAULT_EVAL_INTERVAL,
        ladder_games: int = _DEFAULT_LADDER_GAMES,
        ladder_depths: str = _DEFAULT_LADDER_DEPTHS_CLI,
    ) -> None:
        remote_train = train_remote.with_options(gpu=gpu) if gpu else train_remote
        result = remote_train.remote(
            game=game,
            iterations=iterations,
            self_play_games=self_play_games,
            sims=sims,
            seed=seed,
            gpu=gpu,
            eval_games=eval_games,
            eval_sims=eval_sims,
            gating_interval=gating_interval,
            gating_games=gating_games,
            gating_threshold=gating_threshold,
            eval_interval=eval_interval,
            ladder_games=ladder_games,
            ladder_depths=ladder_depths,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
