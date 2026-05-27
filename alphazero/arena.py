"""Evaluation players and match runner for AlphaZero agents."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.multiprocessing as torch_mp

from alphazero.baselines import NegamaxPlayer
from alphazero.c4_solver import NodeBudgetExceeded, solve as solve_connect_four
from alphazero.game import Game, State
from alphazero.games import GAME_CHOICES, game_from_name
from alphazero.games.connectfour import ConnectFour
from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet
from alphazero.selfplay import SelfPlayExample, play_game
from alphazero.train import (
    ReplayBuffer,
    compute_loss,
    make_optimizer,
    save_checkpoint,
    train_iteration,
)

_WANDB_PROJECT_PREFIX = "alphazero"
DEFAULT_LADDER_DEPTHS = [1, 2, 4, 6]
DEFAULT_ELO = 0.0
ELO_K = 32.0
C4_BOARD_CELLS = 42
DEFAULT_C4_SOLVER_POSITIONS = 8
DEFAULT_C4_SOLVER_MAX_EMPTY_CELLS = 16
DEFAULT_C4_SOLVER_MAX_NODES = 250_000


class WandbRun(Protocol):
    url: str | None

    def log(self, data: Mapping[str, int | float], step: int | None = None) -> None: ...

    def finish(self) -> None: ...


class Player(Protocol):
    def select_action(self, game: Game, state: State) -> int: ...


class RandomPlayer:
    """Uniformly samples legal moves."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game, state: State) -> int:
        legal = game.legal_moves(state)
        if not legal:
            raise ValueError("cannot select an action for a terminal state")
        return int(self.rng.choice(legal))


class PerfectPlayer:
    """Minimax player for small deterministic zero-sum games such as tic-tac-toe."""

    def __init__(self) -> None:
        self._value_cache: dict[tuple[int, State], int] = {}

    def select_action(self, game: Game, state: State) -> int:
        legal = game.legal_moves(state)
        if not legal:
            raise ValueError("cannot select an action for a terminal state")

        scores = {
            action: -self.value(game, game.apply_move(state, action))
            for action in legal
        }
        best_score = max(scores.values())
        return min(action for action, score in scores.items() if score == best_score)

    def value(self, game: Game, state: State) -> int:
        """Minimax value from the current player's perspective."""

        key = (id(game), state)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached

        winner = game.winner(state)
        if winner is not None:
            if winner == 0:
                value = 0
            elif winner == game.current_player(state):
                value = 1
            else:
                value = -1
            self._value_cache[key] = value
            return value

        value = max(
            -self.value(game, game.apply_move(state, action))
            for action in game.legal_moves(state)
        )
        self._value_cache[key] = value
        return value


class MCTSPlayer:
    """MCTS policy player backed by a network with the AlphaZero `predict` API."""

    def __init__(
        self,
        net,
        *,
        c_puct: float = 1.5,
        num_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.0,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game, state: State) -> int:
        mcts = MCTS(
            self.net,
            game,
            c_puct=self.c_puct,
            num_simulations=self.num_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_eps=self.dirichlet_eps,
            seed=int(self.rng.integers(0, np.iinfo(np.int32).max)),
        )
        pi = mcts.run(state, add_noise=False)
        if pi.sum() <= 0:
            legal = game.legal_moves(state)
            if not legal:
                raise ValueError("cannot select an action for a terminal state")
            return int(self.rng.choice(legal))
        return mcts.select_action(pi, temperature=self.temperature, rng=self.rng)


def play_match(
    player_a: Player, player_b: Player, game: Game, n_games: int
) -> tuple[int, int, int]:
    """Play `n_games`, alternating seats, and return (wins_a, draws, wins_b)."""

    if n_games <= 0:
        raise ValueError("n_games must be positive")

    wins_a = 0
    draws = 0
    wins_b = 0
    for game_index in range(n_games):
        state = game.initial_state()
        player_a_mark = 1 if game_index % 2 == 0 else -1

        while not game.is_terminal(state):
            current_player = game.current_player(state)
            player = player_a if current_player == player_a_mark else player_b
            action = player.select_action(game, state)
            if action not in game.legal_moves(state):
                raise ValueError(f"player selected illegal action {action}")
            state = game.apply_move(state, action)

        winner = game.winner(state)
        if winner == 0:
            draws += 1
        elif winner == player_a_mark:
            wins_a += 1
        else:
            wins_b += 1

    return wins_a, draws, wins_b


def update_elo(
    rating: float,
    opponent_rating: float,
    score: float,
    *,
    k: float = ELO_K,
) -> float:
    """Return an Elo rating updated from one aggregate match score."""

    expected = 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))
    return rating + k * (score - expected)


def gating_match(
    candidate: Player,
    best: Player,
    game: Game,
    *,
    n_games: int,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate whether a candidate should replace the current best player."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    wins, draws, losses = play_match(candidate, best, game, n_games)
    decisive_games = wins + losses
    gating_winrate = wins / decisive_games if decisive_games > 0 else 0.0
    match_score = (wins + 0.5 * draws) / n_games
    promoted = int(gating_winrate >= threshold)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate": gating_winrate,
        "score": match_score,
        "promoted": promoted,
    }


def evaluate_ladder(
    player: Player,
    game: Game,
    *,
    n_games: int,
    negamax_depths: Sequence[int] = DEFAULT_LADDER_DEPTHS,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate a player against random play and negamax depth baselines."""

    random_wins, _, _ = play_match(
        player,
        RandomPlayer(seed=seed),
        game,
        n_games,
    )
    metrics = {"eval/ladder_random_winrate": random_wins / n_games}
    for depth in negamax_depths:
        if depth < 1:
            raise ValueError("negamax depths must be at least 1")
        wins, _, _ = play_match(player, NegamaxPlayer(depth=depth), game, n_games)
        metrics[f"eval/ladder_negamax_d{depth}_winrate"] = wins / n_games
    return metrics


def evaluate_connect_four_solver_anchor(
    net,
    game: ConnectFour,
    *,
    n_positions: int = DEFAULT_C4_SOLVER_POSITIONS,
    max_empty_cells: int = DEFAULT_C4_SOLVER_MAX_EMPTY_CELLS,
    solver_max_nodes: int = DEFAULT_C4_SOLVER_MAX_NODES,
    seed: int = 0,
    positions: Sequence[State] | None = None,
    player: Player | None = None,
) -> dict[str, float]:
    """Compare a Connect Four agent against exact solver labels.

    The value metric uses ``net.predict(game.encode(state))``. The move metric
    uses ``player`` when provided; otherwise it uses the net policy argmax over
    legal moves.
    """

    if not isinstance(game, ConnectFour):
        raise TypeError("evaluate_connect_four_solver_anchor requires ConnectFour")
    if n_positions <= 0:
        raise ValueError("n_positions must be positive")
    if not 1 <= max_empty_cells <= C4_BOARD_CELLS:
        raise ValueError("max_empty_cells must be in [1, 42]")
    if solver_max_nodes <= 0:
        raise ValueError("solver_max_nodes must be positive")

    eval_positions = (
        list(positions)
        if positions is not None
        else _sample_connect_four_solver_positions(
            game,
            n_positions=n_positions,
            max_empty_cells=max_empty_cells,
            seed=seed,
        )
    )
    value_errors: list[float] = []
    policy_matches = 0
    blunders = 0

    for state in eval_positions:
        if game.is_terminal(state):
            continue
        legal = game.legal_moves(state)
        if not legal:
            continue
        try:
            solver_value, optimal_moves = solve_connect_four(
                state,
                max_nodes=solver_max_nodes,
            )
        except NodeBudgetExceeded:
            continue
        if not optimal_moves:
            continue

        policy, net_value = net.predict(game.encode(state))
        chosen_move = (
            player.select_action(game, state)
            if player is not None
            else _policy_argmax_legal(policy, legal)
        )
        if chosen_move not in legal:
            raise ValueError(f"agent selected illegal action {chosen_move}")

        try:
            child_value, _ = solve_connect_four(
                game.apply_move(state, chosen_move),
                max_nodes=solver_max_nodes,
            )
        except NodeBudgetExceeded:
            continue

        value_errors.append(abs(float(net_value) - float(solver_value)))
        if chosen_move in optimal_moves:
            policy_matches += 1
        if -child_value < solver_value:
            blunders += 1

    count = len(value_errors)
    if count == 0:
        return {
            "eval/c4_value_mae": 0.0,
            "eval/c4_policy_match": 0.0,
            "eval/c4_blunder_rate": 0.0,
            "eval/c4_solver_positions": 0.0,
        }
    return {
        "eval/c4_value_mae": float(np.mean(value_errors)),
        "eval/c4_policy_match": policy_matches / count,
        "eval/c4_blunder_rate": blunders / count,
        "eval/c4_solver_positions": float(count),
    }


def _sample_connect_four_solver_positions(
    game: ConnectFour,
    *,
    n_positions: int,
    max_empty_cells: int,
    seed: int,
) -> list[State]:
    rng = np.random.default_rng(seed)
    target_moves = C4_BOARD_CELLS - max_empty_cells
    positions: list[State] = []
    max_attempts = max(n_positions * 50, 50)

    for _ in range(max_attempts):
        state = game.initial_state()
        for _ in range(target_moves):
            legal = game.legal_moves(state)
            if not legal or game.is_terminal(state):
                break
            state = game.apply_move(state, int(rng.choice(legal)))
        if not game.is_terminal(state) and game.legal_moves(state):
            positions.append(state)
            if len(positions) == n_positions:
                break

    return positions


def _policy_argmax_legal(policy: np.ndarray, legal: Sequence[int]) -> int:
    scores = np.asarray(policy, dtype=np.float64)
    return max(legal, key=lambda action: (float(scores[action]), -action))


def immediate_winning_moves(game: Game, state: State) -> list[int]:
    """Return legal moves that win immediately for the player to move."""

    player = game.current_player(state)
    return [
        action
        for action in game.legal_moves(state)
        if game.winner(game.apply_move(state, action)) == player
    ]


def immediate_blocking_moves(game: Game, state: State) -> list[int]:
    """Return legal moves that remove every opponent one-ply win.

    A position only counts as a block tactic if at least one legal move would
    leave the opponent with an immediate win. Winning immediately is also a
    valid block because the opponent never gets the threatened reply.
    """

    legal = game.legal_moves(state)
    if not legal:
        return []

    safe_moves: list[int] = []
    unsafe_found = False
    for action in legal:
        next_state = game.apply_move(state, action)
        if game.is_terminal(next_state):
            safe_moves.append(action)
            continue
        if immediate_winning_moves(game, next_state):
            unsafe_found = True
        else:
            safe_moves.append(action)

    return safe_moves if unsafe_found else []


def tactical_action_rate(
    player: Player,
    game: Game,
    positions: Sequence[State],
    target_moves: Callable[[Game, State], list[int]],
) -> float:
    """Score how often `player` selects one of the exact tactical targets."""

    hits = 0
    total = 0
    for state in positions:
        targets = target_moves(game, state)
        if not targets:
            continue
        action = player.select_action(game, state)
        if action not in game.legal_moves(state):
            raise ValueError(f"player selected illegal action {action}")
        total += 1
        if action in targets:
            hits += 1

    if total == 0:
        raise ValueError("no tactical positions with target moves")
    return hits / total


def connect_four_tactical_positions(
    game: ConnectFour | None = None,
) -> tuple[list[State], list[State]]:
    """Return fixed Connect Four positions for one-ply wins and blocks."""

    c4 = game if game is not None else ConnectFour()
    win_sequences = (
        (0, 0, 1, 1, 2, 2),  # horizontal
        (0, 1, 0, 1, 0, 2),  # vertical
        (0, 1, 1, 2, 3, 2, 2, 3, 4, 3),  # diagonal up-right
        (6, 5, 5, 4, 3, 4, 4, 3, 2, 3),  # diagonal down-right
    )
    block_sequences = (
        (2, 0, 4, 0, 6, 0),  # vertical threat
        (6, 0, 6, 1, 5, 2),  # horizontal threat
    )
    return (
        [_state_after_moves(c4, moves) for moves in win_sequences],
        [_state_after_moves(c4, moves) for moves in block_sequences],
    )


def evaluate_connect_four_tactics(
    player: Player,
    game: ConnectFour | None = None,
) -> dict[str, float]:
    """Evaluate exact one-ply Connect Four win and block tactics."""

    c4 = game if game is not None else ConnectFour()
    win_positions, block_positions = connect_four_tactical_positions(c4)
    return {
        "immediate_win_rate": tactical_action_rate(
            player, c4, win_positions, immediate_winning_moves
        ),
        "block_rate": tactical_action_rate(
            player, c4, block_positions, immediate_blocking_moves
        ),
    }


def _state_after_moves(game: Game, moves: Sequence[int]) -> State:
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return state


def opening_temperature_schedule(move_index: int) -> float:
    """Explore early openings, then play greedily from MCTS visits."""

    return 1.0 if move_index < 3 else 0.0


def _self_play_cfg(
    cfg: Mapping[str, object] | None,
    seed: int,
) -> dict[str, object]:
    merged: dict[str, object] = {
        "num_simulations": 64,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25,
    }
    if cfg is not None:
        merged.update(cfg)
    if float(merged.get("dirichlet_eps", 0.0)) <= 0.0:
        merged["dirichlet_eps"] = 0.25
    if float(merged.get("dirichlet_alpha", 0.0)) <= 0.0:
        merged["dirichlet_alpha"] = 0.3
    merged["seed"] = seed
    return merged


def _mcts_player(
    net,
    cfg: Mapping[str, object] | None,
    *,
    seed: int,
) -> MCTSPlayer:
    cfg = cfg or {}
    return MCTSPlayer(
        net,
        c_puct=float(cfg.get("c_puct", 1.5)),
        num_simulations=int(cfg.get("num_simulations", 100)),
        dirichlet_alpha=float(cfg.get("dirichlet_alpha", 0.3)),
        dirichlet_eps=float(cfg.get("dirichlet_eps", 0.0)),
        temperature=float(cfg.get("temperature", 0.0)),
        seed=seed,
    )


def _default_wandb_project(game_name: str) -> str:
    return f"{_WANDB_PROJECT_PREFIX}-{game_name}"


def _wandb_project_for_game(game: Game) -> str:
    if isinstance(game, TicTacToe):
        return _default_wandb_project("tictactoe")
    if isinstance(game, ConnectFour):
        return _default_wandb_project("connectfour")
    return _default_wandb_project(type(game).__name__.lower())


def _checkpoint_game_name(game: Game) -> str:
    known_names = {
        "TicTacToe": "tictactoe",
        "ConnectFour": "connectfour",
        "Gomoku": "gomoku",
        "Go": "go",
    }
    return known_names.get(type(game).__name__, type(game).__name__.lower())


def _periodic_checkpoint_path(
    checkpoint_dir: str | Path,
    game: Game,
    iteration_number: int,
) -> Path:
    return (
        Path(checkpoint_dir)
        / _checkpoint_game_name(game)
        / f"iter_{iteration_number:04d}.pt"
    )


def _clone_net(net: AlphaZeroNet) -> AlphaZeroNet:
    clone = AlphaZeroNet(net.num_planes, net.board_shape, net.action_size)
    clone.load_state_dict(net.state_dict())
    clone.train(False)
    return clone


def _worker_net_kwargs(net: AlphaZeroNet) -> dict[str, object]:
    return {
        "num_planes": net.num_planes,
        "board_shape": net.board_shape,
        "action_size": net.action_size,
        "channels": net.channels,
        "num_res_blocks": net.num_res_blocks,
    }


def _worker_state_dict(net: AlphaZeroNet) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()
    }


_SELF_PLAY_WORKER_PAYLOAD: Any | None = None
_SELF_PLAY_WORKER_GENERATION: int | None = None
_SELF_PLAY_WORKER_NET: AlphaZeroNet | None = None
_SELF_PLAY_WORKER_GAME: Game | None = None
_SELF_PLAY_WORKER_MCTS_CFG: dict[str, object] | None = None


def _init_self_play_worker(shared_payload: Any) -> None:
    global _SELF_PLAY_WORKER_PAYLOAD
    torch.set_num_threads(1)
    _SELF_PLAY_WORKER_PAYLOAD = shared_payload


def _load_self_play_worker_generation(generation: int) -> None:
    global _SELF_PLAY_WORKER_GENERATION
    global _SELF_PLAY_WORKER_GAME
    global _SELF_PLAY_WORKER_MCTS_CFG
    global _SELF_PLAY_WORKER_NET

    if _SELF_PLAY_WORKER_GENERATION == generation:
        return
    if _SELF_PLAY_WORKER_PAYLOAD is None:
        raise RuntimeError("self-play worker was not initialized")

    payload = _SELF_PLAY_WORKER_PAYLOAD
    if int(payload.generation) != generation:
        raise RuntimeError(
            f"self-play worker expected generation {generation}, "
            f"got {payload.generation}"
        )

    net_kwargs = dict(cast(Mapping[str, object], payload.net_kwargs))
    net = AlphaZeroNet(
        int(net_kwargs["num_planes"]),
        cast(tuple[int, int], net_kwargs["board_shape"]),
        int(net_kwargs["action_size"]),
        channels=int(net_kwargs["channels"]),
        num_res_blocks=int(net_kwargs["num_res_blocks"]),
    )
    net.load_state_dict(cast(dict[str, torch.Tensor], payload.state_dict))
    net.train(False)

    _SELF_PLAY_WORKER_GENERATION = generation
    _SELF_PLAY_WORKER_GAME = cast(Game, payload.game)
    _SELF_PLAY_WORKER_MCTS_CFG = dict(
        cast(Mapping[str, object], payload.self_play_mcts_cfg)
    )
    _SELF_PLAY_WORKER_NET = net


def _self_play_one_game_worker(
    task: tuple[int, int, int, bool],
) -> tuple[int, list[SelfPlayExample], list[tuple[str, float]]]:
    game_index, game_seed, generation, collect_timings = task
    _load_self_play_worker_generation(generation)
    if (
        _SELF_PLAY_WORKER_NET is None
        or _SELF_PLAY_WORKER_GAME is None
        or _SELF_PLAY_WORKER_MCTS_CFG is None
    ):
        raise RuntimeError("self-play worker has no active payload")

    timing_events: list[tuple[str, float]] = []

    def record_timing(name: str, seconds: float) -> None:
        timing_events.append((name, seconds))

    timing_hook = record_timing if collect_timings else None
    examples = play_game(
        _SELF_PLAY_WORKER_NET,
        _SELF_PLAY_WORKER_GAME,
        _self_play_cfg(_SELF_PLAY_WORKER_MCTS_CFG, game_seed),
        temperature_schedule=opening_temperature_schedule,
        timing_hook=timing_hook,
    )
    return game_index, examples, timing_events


class _SelfPlayWorkerPool:
    def __init__(self, worker_count: int) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")

        self.worker_count = worker_count
        self._ctx = torch_mp.get_context("spawn")
        self._manager = self._ctx.Manager()
        self._payload = self._manager.Namespace()
        self._payload.generation = 0
        self._generation = 0
        self._closed = False
        self._pool = self._ctx.Pool(
            processes=worker_count,
            initializer=_init_self_play_worker,
            initargs=(self._payload,),
        )

    def __enter__(self) -> _SelfPlayWorkerPool:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close(terminate=exc_type is not None)
        return False

    def close(self, *, terminate: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if terminate:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
        finally:
            self._manager.shutdown()

    def run_iteration(
        self,
        net: AlphaZeroNet,
        game: Game,
        self_play_mcts_cfg: Mapping[str, object] | None,
        indexed_game_seeds: Sequence[tuple[int, int]],
        *,
        collect_timings: bool,
    ) -> tuple[
        dict[int, list[SelfPlayExample]],
        dict[int, list[tuple[str, float]]],
    ]:
        self._generation += 1
        self._payload.net_kwargs = _worker_net_kwargs(net)
        self._payload.state_dict = _worker_state_dict(net)
        self._payload.game = game
        self._payload.self_play_mcts_cfg = dict(self_play_mcts_cfg or {})
        self._payload.generation = self._generation

        tasks = [
            (game_index, game_seed, self._generation, collect_timings)
            for game_index, game_seed in indexed_game_seeds
        ]
        by_game_index: dict[int, list[SelfPlayExample]] = {}
        timings_by_game_index: dict[int, list[tuple[str, float]]] = {}
        for game_index, game_examples, timing_events in self._pool.imap_unordered(
            _self_play_one_game_worker,
            tasks,
            chunksize=1,
        ):
            by_game_index[game_index] = game_examples
            timings_by_game_index[game_index] = timing_events

        return by_game_index, timings_by_game_index


def _self_play_examples_for_iteration(
    net: AlphaZeroNet,
    game: Game,
    self_play_mcts_cfg: Mapping[str, object] | None,
    game_seeds: Sequence[int],
    *,
    n_selfplay_workers: int,
    timing_hook: Callable[[str, float], None] | None = None,
    worker_pool: _SelfPlayWorkerPool | None = None,
) -> list[SelfPlayExample]:
    if n_selfplay_workers <= 0:
        raise ValueError("n_selfplay_workers must be positive")

    indexed_game_seeds = [(index, int(seed)) for index, seed in enumerate(game_seeds)]
    if n_selfplay_workers == 1:
        examples: list[SelfPlayExample] = []
        for _, game_seed in indexed_game_seeds:
            play_kwargs: dict[str, Any] = {}
            if timing_hook is not None:
                play_kwargs["timing_hook"] = timing_hook
            examples.extend(
                play_game(
                    net,
                    game,
                    _self_play_cfg(self_play_mcts_cfg, game_seed),
                    temperature_schedule=opening_temperature_schedule,
                    **play_kwargs,
                )
            )
        return examples

    if not indexed_game_seeds:
        return []

    worker_count = min(n_selfplay_workers, len(indexed_game_seeds))
    owns_worker_pool = worker_pool is None
    if worker_pool is None:
        worker_pool = _SelfPlayWorkerPool(worker_count)
    elif worker_pool.worker_count != worker_count:
        raise ValueError(
            "worker_pool worker_count must match the requested self-play workers"
        )

    try:
        by_game_index, timings_by_game_index = worker_pool.run_iteration(
            net,
            game,
            self_play_mcts_cfg,
            indexed_game_seeds,
            collect_timings=timing_hook is not None,
        )
    except BaseException:
        if owns_worker_pool:
            worker_pool.close(terminate=True)
        raise
    else:
        if owns_worker_pool:
            worker_pool.close()

    examples = []
    for game_index, _ in indexed_game_seeds:
        examples.extend(by_game_index[game_index])
        if timing_hook is not None:
            for name, seconds in timings_by_game_index[game_index]:
                timing_hook(name, seconds)
    return examples


def _training_run_config(
    *,
    iterations: int,
    self_play_games_per_iteration: int,
    self_play_mcts_cfg: Mapping[str, object] | None,
    replay_capacity: int,
    batch_size: int,
    epochs: int,
    lr: float,
    l2_reg: float,
    checkpoint_path: str | Path | None,
    checkpoint_every: int | None,
    checkpoint_dir: str | Path,
    seed: int,
    gating_interval: int = 5,
    gating_games: int = 20,
    gating_threshold: float = 0.55,
    eval_interval: int = 5,
    ladder_games: int = 20,
    ladder_depths: Sequence[int] = DEFAULT_LADDER_DEPTHS,
    n_selfplay_workers: int = 1,
) -> dict[str, object]:
    mcts_cfg = _self_play_cfg(self_play_mcts_cfg, seed)
    return {
        "iterations": iterations,
        "self_play_games": self_play_games_per_iteration,
        "self_play_sims": mcts_cfg["num_simulations"],
        "c_puct": mcts_cfg["c_puct"],
        "dirichlet_alpha": mcts_cfg["dirichlet_alpha"],
        "dirichlet_eps": mcts_cfg["dirichlet_eps"],
        "replay_capacity": replay_capacity,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "l2_reg": l2_reg,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "checkpoint_every": checkpoint_every or 0,
        "checkpoint_dir": str(checkpoint_dir),
        "seed": seed,
        "gating_interval": gating_interval,
        "gating_games": gating_games,
        "gating_threshold": gating_threshold,
        "eval_interval": eval_interval,
        "ladder_games": ladder_games,
        "ladder_depths": list(ladder_depths),
        "self_play_workers": n_selfplay_workers,
    }


def _init_wandb(
    enabled: bool,
    *,
    project: str,
    run_name: str | None,
    config: Mapping[str, object],
) -> WandbRun | None:
    if not enabled:
        return None

    try:
        wandb_module = __import__("wandb")
        init = getattr(wandb_module, "init")
        return cast(WandbRun, init(project=project, name=run_name, config=dict(config)))
    except Exception as exc:
        print(f"Warning: wandb disabled: {exc}", file=sys.stderr)
        return None


def _print_wandb_url(run: WandbRun | None) -> None:
    if run is None:
        return
    url = getattr(run, "url", None)
    if url:
        print(f"wandb run: {url}")


def _wandb_log(
    run: WandbRun | None,
    metrics: Mapping[str, float | int | str],
    *,
    step: int,
) -> None:
    if run is None:
        return

    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if not numeric_metrics:
        return

    try:
        run.log(numeric_metrics, step=step)
    except Exception as exc:
        print(f"Warning: wandb log skipped: {exc}", file=sys.stderr)


def _print_iteration_progress(
    iteration: int, total: int, metrics: Mapping[str, object]
) -> None:
    """Emit a one-line per-iteration progress summary to stdout.

    Independent of wandb, so local and --no-wandb runs still show progress.
    """

    parts = [f"iter {iteration}/{total}"]
    for key, label, fmt in (
        ("loss", "loss", "{:.3f}"),
        ("eval/elo", "elo", "{:.1f}"),
        ("eval/gating_winrate", "gate", "{:.2f}"),
        ("self_play_games_per_sec", "games/s", "{:.2f}"),
    ):
        value = metrics.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            parts.append(f"{label} {fmt.format(float(value))}")
    print(" | ".join(parts), flush=True)


def _wandb_finish(run: WandbRun | None) -> None:
    if run is None:
        return

    try:
        run.finish()
    except Exception as exc:
        print(f"Warning: wandb finish skipped: {exc}", file=sys.stderr)


def train_agent(
    game: Game,
    *,
    iterations: int = 25,
    self_play_games_per_iteration: int = 8,
    self_play_mcts_cfg: Mapping[str, object] | None = None,
    eval_mcts_cfg: Mapping[str, object] | None = None,
    replay_capacity: int = 8192,
    batch_size: int = 64,
    epochs: int = 2,
    lr: float = 5e-3,
    l2_reg: float = 1e-5,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    seed: int = 0,
    wandb_enabled: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_run: WandbRun | None = None,
    wandb_config: Mapping[str, object] | None = None,
    gating_interval: int = 5,
    gating_games: int = 20,
    gating_threshold: float = 0.55,
    eval_interval: int = 5,
    ladder_games: int = 20,
    ladder_depths: Sequence[int] = DEFAULT_LADDER_DEPTHS,
    n_selfplay_workers: int = 1,
    timing_hook: Callable[[str, float], None] | None = None,
) -> tuple[AlphaZeroNet, dict[str, float | int | str]]:
    """Train an AlphaZero agent for `game` from tabula-rasa self-play only."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if self_play_games_per_iteration <= 0:
        raise ValueError("self_play_games_per_iteration must be positive")
    if gating_interval <= 0:
        raise ValueError("gating_interval must be positive")
    if gating_games <= 0:
        raise ValueError("gating_games must be positive")
    if not 0.0 <= gating_threshold <= 1.0:
        raise ValueError("gating_threshold must be in [0, 1]")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if ladder_games <= 0:
        raise ValueError("ladder_games must be positive")
    if any(depth < 1 for depth in ladder_depths):
        raise ValueError("ladder_depths must all be at least 1")
    if n_selfplay_workers <= 0:
        raise ValueError("n_selfplay_workers must be positive")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive when set")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AlphaZeroNet(game.num_planes, game.board_shape, game.action_size)
    reference_net = _clone_net(net)
    reference_elo = DEFAULT_ELO
    last_gating_winrate = 0.0
    optimizer = make_optimizer(net, optimizer_name="adam", lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)
    evaluation_mcts_cfg = (
        eval_mcts_cfg if eval_mcts_cfg is not None else self_play_mcts_cfg
    )
    run_config = _training_run_config(
        iterations=iterations,
        self_play_games_per_iteration=self_play_games_per_iteration,
        self_play_mcts_cfg=self_play_mcts_cfg,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        l2_reg=l2_reg,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        gating_interval=gating_interval,
        gating_games=gating_games,
        gating_threshold=gating_threshold,
        eval_interval=eval_interval,
        ladder_games=ladder_games,
        ladder_depths=ladder_depths,
        n_selfplay_workers=n_selfplay_workers,
    )
    run_config["game"] = type(game).__name__
    if wandb_config is not None:
        run_config.update(wandb_config)

    active_wandb_run = wandb_run
    owns_wandb_run = False
    if active_wandb_run is None:
        active_wandb_run = _init_wandb(
            wandb_enabled,
            project=wandb_project or _wandb_project_for_game(game),
            run_name=wandb_run_name,
            config=run_config,
        )
        owns_wandb_run = active_wandb_run is not None

    metrics: dict[str, float | int | str] = {}
    self_play_worker_pool: _SelfPlayWorkerPool | None = None
    terminate_self_play_pool = False
    try:
        if n_selfplay_workers > 1:
            self_play_worker_pool = _SelfPlayWorkerPool(
                min(n_selfplay_workers, self_play_games_per_iteration)
            )

        for iteration in range(iterations):
            iteration_number = iteration + 1
            iteration_started = time.perf_counter()
            game_seeds = [
                int(rng.integers(0, np.iinfo(np.int32).max))
                for _ in range(self_play_games_per_iteration)
            ]
            examples = _self_play_examples_for_iteration(
                net,
                game,
                self_play_mcts_cfg,
                game_seeds,
                n_selfplay_workers=n_selfplay_workers,
                timing_hook=timing_hook,
                worker_pool=self_play_worker_pool,
            )
            loss_before, _ = compute_loss(net, examples, l2_reg=l2_reg)
            metrics = train_iteration(
                net,
                examples,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                epochs=epochs,
                l2_reg=l2_reg,
                shuffle=True,
                rng=rng,
                timing_hook=timing_hook,
            )
            loss_after, _ = compute_loss(net, examples, l2_reg=l2_reg)
            metrics["iteration"] = iteration_number
            metrics["self_play_examples"] = len(examples)
            metrics["loss_before"] = float(loss_before.detach().cpu().item())
            metrics["loss_after"] = float(loss_after.detach().cpu().item())
            metrics["loss_delta"] = float(metrics["loss_before"]) - float(
                metrics["loss_after"]
            )
            iteration_seconds = max(time.perf_counter() - iteration_started, 1e-12)
            metrics["iteration_seconds"] = iteration_seconds
            metrics["iters_per_sec"] = 1.0 / iteration_seconds
            metrics["self_play_games_per_sec"] = (
                self_play_games_per_iteration / iteration_seconds
            )

            promoted = 0
            if iteration_number % gating_interval == 0:
                candidate_net = _clone_net(net)
                gate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                gate = gating_match(
                    _mcts_player(
                        candidate_net,
                        evaluation_mcts_cfg,
                        seed=gate_seed,
                    ),
                    _mcts_player(
                        reference_net,
                        evaluation_mcts_cfg,
                        seed=gate_seed + 1,
                    ),
                    game,
                    n_games=gating_games,
                    threshold=gating_threshold,
                )
                last_gating_winrate = float(gate["winrate"])
                promoted = int(gate["promoted"])
                metrics["eval/gating_wins"] = int(gate["wins"])
                metrics["eval/gating_draws"] = int(gate["draws"])
                metrics["eval/gating_losses"] = int(gate["losses"])
                metrics["eval/gating_score"] = float(gate["score"])
                if promoted:
                    reference_elo = update_elo(
                        reference_elo,
                        reference_elo,
                        float(gate["score"]),
                    )
                    reference_net = candidate_net

            metrics["eval/elo"] = reference_elo
            metrics["eval/gating_winrate"] = last_gating_winrate
            metrics["eval/promoted"] = promoted

            if iteration_number % eval_interval == 0:
                ladder_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                metrics.update(
                    evaluate_ladder(
                        _mcts_player(net, evaluation_mcts_cfg, seed=ladder_seed),
                        game,
                        n_games=ladder_games,
                        negamax_depths=ladder_depths,
                        seed=ladder_seed + 1,
                    )
                )
                if isinstance(game, ConnectFour):
                    metrics.update(
                        evaluate_connect_four_solver_anchor(
                            net,
                            game,
                            n_positions=DEFAULT_C4_SOLVER_POSITIONS,
                            seed=ladder_seed + 2,
                        )
                    )

            if (
                checkpoint_every is not None
                and iteration_number % checkpoint_every == 0
            ):
                periodic_path = _periodic_checkpoint_path(
                    checkpoint_dir,
                    game,
                    iteration_number,
                )
                save_checkpoint(
                    net,
                    periodic_path,
                    optimizer=optimizer,
                    metrics=metrics,
                )
                metrics["checkpoint/periodic_path"] = str(periodic_path)

            _wandb_log(active_wandb_run, metrics, step=iteration + 1)
            _print_iteration_progress(iteration_number, iterations, metrics)
    except BaseException:
        terminate_self_play_pool = True
        raise
    finally:
        if self_play_worker_pool is not None:
            self_play_worker_pool.close(terminate=terminate_self_play_pool)
        if owns_wandb_run:
            _wandb_finish(active_wandb_run)

    if checkpoint_path is not None:
        save_checkpoint(
            net,
            checkpoint_path,
            optimizer=optimizer,
            metrics=metrics,
        )
        metrics["checkpoint_path"] = str(checkpoint_path)
    return net, metrics


def train_tictactoe_agent(
    *,
    iterations: int = 25,
    self_play_games_per_iteration: int = 8,
    self_play_mcts_cfg: Mapping[str, object] | None = None,
    eval_mcts_cfg: Mapping[str, object] | None = None,
    replay_capacity: int = 8192,
    batch_size: int = 64,
    epochs: int = 2,
    lr: float = 5e-3,
    l2_reg: float = 1e-5,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    seed: int = 0,
    wandb_enabled: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_run: WandbRun | None = None,
    wandb_config: Mapping[str, object] | None = None,
    gating_interval: int = 5,
    gating_games: int = 20,
    gating_threshold: float = 0.55,
    eval_interval: int = 5,
    ladder_games: int = 20,
    ladder_depths: Sequence[int] = DEFAULT_LADDER_DEPTHS,
    n_selfplay_workers: int = 1,
    timing_hook: Callable[[str, float], None] | None = None,
) -> tuple[AlphaZeroNet, dict[str, float | int | str]]:
    """Train a compact tic-tac-toe agent from tabula-rasa self-play only."""

    return train_agent(
        TicTacToe(),
        iterations=iterations,
        self_play_games_per_iteration=self_play_games_per_iteration,
        self_play_mcts_cfg=self_play_mcts_cfg,
        eval_mcts_cfg=eval_mcts_cfg,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        l2_reg=l2_reg,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_run=wandb_run,
        wandb_config=wandb_config,
        gating_interval=gating_interval,
        gating_games=gating_games,
        gating_threshold=gating_threshold,
        eval_interval=eval_interval,
        ladder_games=ladder_games,
        ladder_depths=ladder_depths,
        n_selfplay_workers=n_selfplay_workers,
        timing_hook=timing_hook,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an AlphaZero agent."
    )
    parser.add_argument(
        "--game",
        choices=GAME_CHOICES,
        default="tictactoe",
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-play-games", type=int, default=24)
    parser.add_argument("--self-play-sims", type=int, default=128)
    parser.add_argument(
        "--self-play-workers",
        type=int,
        default=1,
        help="Number of spawned worker processes for self-play games.",
    )
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument(
        "--mcts-batch-size",
        type=int,
        default=16,
        help="Leaf-parallel self-play MCTS batch size (1 = sequential search).",
    )
    parser.add_argument("--replay-capacity", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--l2-reg", type=float, default=1e-5)
    parser.add_argument("--gating-interval", type=int, default=5)
    parser.add_argument("--gating-games", type=int, default=20)
    parser.add_argument("--gating-threshold", type=float, default=0.55)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--ladder-games", type=int, default=20)
    parser.add_argument(
        "--ladder-depths",
        type=int,
        nargs="+",
        default=DEFAULT_LADDER_DEPTHS,
    )
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save checkpoints/<game>/iter_NNNN.pt every N iterations.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory for periodic per-game checkpoints.",
    )
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", default=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)

    if args.checkpoint_every is not None and args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive when set")

    game = game_from_name(args.game)
    wandb_project = args.wandb_project or _default_wandb_project(args.game)
    checkpoint = args.checkpoint or Path(f"checkpoints/{args.game}.pt")
    self_play_mcts_cfg = {
        "num_simulations": args.self_play_sims,
        "dirichlet_eps": args.dirichlet_eps,
        "batch_size": args.mcts_batch_size,
    }
    run_config = _training_run_config(
        iterations=args.iterations,
        self_play_games_per_iteration=args.self_play_games,
        self_play_mcts_cfg=self_play_mcts_cfg,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        l2_reg=args.l2_reg,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        gating_interval=args.gating_interval,
        gating_games=args.gating_games,
        gating_threshold=args.gating_threshold,
        eval_interval=args.eval_interval,
        ladder_games=args.ladder_games,
        ladder_depths=args.ladder_depths,
        n_selfplay_workers=args.self_play_workers,
    )
    run_config.update(
        {
            "game": args.game,
            "wandb_project": wandb_project,
            "eval_games": args.eval_games,
            "eval_sims": args.eval_sims,
        }
    )
    wandb_run = _init_wandb(
        args.wandb,
        project=wandb_project,
        run_name=args.wandb_run_name,
        config=run_config,
    )
    _print_wandb_url(wandb_run)
    try:
        net, metrics = train_agent(
            game,
            iterations=args.iterations,
            self_play_games_per_iteration=args.self_play_games,
            self_play_mcts_cfg=self_play_mcts_cfg,
            eval_mcts_cfg={"num_simulations": args.eval_sims},
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            l2_reg=args.l2_reg,
            checkpoint_path=checkpoint,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            wandb_run=wandb_run,
            wandb_config=run_config,
            gating_interval=args.gating_interval,
            gating_games=args.gating_games,
            gating_threshold=args.gating_threshold,
            eval_interval=args.eval_interval,
            ladder_games=args.ladder_games,
            ladder_depths=args.ladder_depths,
            n_selfplay_workers=args.self_play_workers,
        )

        if args.game == "connectfour":
            agent = MCTSPlayer(net, num_simulations=args.eval_sims, seed=args.seed)
            tactical_metrics = evaluate_connect_four_tactics(
                agent, cast(ConnectFour, game)
            )
            solver_metrics = evaluate_connect_four_solver_anchor(
                net,
                cast(ConnectFour, game),
                n_positions=max(1, min(args.eval_games, DEFAULT_C4_SOLVER_POSITIONS)),
                seed=args.seed,
            )
            random_wins, random_draws, random_losses = play_match(
                agent,
                RandomPlayer(seed=args.seed),
                game,
                args.eval_games,
            )
            vs_random = {
                "wins": random_wins,
                "draws": random_draws,
                "losses": random_losses,
            }
            _wandb_log(
                wandb_run,
                {
                    "eval/c4_immediate_win_rate": tactical_metrics[
                        "immediate_win_rate"
                    ],
                    "eval/c4_block_rate": tactical_metrics["block_rate"],
                    "eval/c4_random_wins": random_wins,
                    "eval/c4_random_draws": random_draws,
                    "eval/c4_random_losses": random_losses,
                    **solver_metrics,
                },
                step=args.iterations,
            )
            print(
                {
                    "metrics": metrics,
                    "c4_tactics": tactical_metrics,
                    "c4_solver": solver_metrics,
                    "vs_random": vs_random,
                }
            )
            return 0

        if args.game == "tictactoe":
            # Tic-tac-toe is the only game with a tractable perfect (minimax)
            # player to benchmark against.
            perfect_wins, perfect_draws, perfect_losses = play_match(
                MCTSPlayer(net, num_simulations=args.eval_sims, seed=args.seed),
                PerfectPlayer(),
                game,
                args.eval_games,
            )
            random_wins, random_draws, random_losses = play_match(
                MCTSPlayer(net, num_simulations=args.eval_sims, seed=args.seed + 1),
                RandomPlayer(seed=args.seed),
                game,
                args.eval_games,
            )
            _wandb_log(
                wandb_run,
                {
                    "eval/perfect_wins": perfect_wins,
                    "eval/perfect_draws": perfect_draws,
                    "eval/perfect_losses": perfect_losses,
                    "eval/random_wins": random_wins,
                    "eval/random_draws": random_draws,
                    "eval/random_losses": random_losses,
                },
                step=args.iterations,
            )
            print(
                {
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
                }
            )
            return 0 if perfect_losses == 0 else 1

        # Games without a tractable perfect player (gomoku, go, ...): benchmark
        # against random play. Gating and the negamax ladder already ran inside
        # train_agent during training.
        random_wins, random_draws, random_losses = play_match(
            MCTSPlayer(net, num_simulations=args.eval_sims, seed=args.seed),
            RandomPlayer(seed=args.seed),
            game,
            args.eval_games,
        )
        _wandb_log(
            wandb_run,
            {
                "eval/random_wins": random_wins,
                "eval/random_draws": random_draws,
                "eval/random_losses": random_losses,
            },
            step=args.iterations,
        )
        print(
            {
                "metrics": metrics,
                "vs_random": {
                    "wins": random_wins,
                    "draws": random_draws,
                    "losses": random_losses,
                },
            }
        )
        return 0
    finally:
        _wandb_finish(wandb_run)


if __name__ == "__main__":
    raise SystemExit(main())
