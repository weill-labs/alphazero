from __future__ import annotations

import numpy as np
import torch

import alphazero.arena as arena
from alphazero.arena import evaluate_connect_four_solver_anchor, train_agent
from alphazero.c4_solver import solve
from alphazero.game import Game, State
from alphazero.games.connectfour import ConnectFour


_DRAW_LINE = [
    3,
    3,
    0,
    2,
    5,
    4,
    5,
    6,
    4,
    2,
    2,
    3,
    3,
    5,
    5,
    2,
    2,
    4,
    3,
    3,
    1,
    1,
    2,
    1,
    5,
    5,
    0,
    4,
    0,
    4,
    4,
    6,
    6,
    6,
    6,
    0,
    1,
    0,
]


_FORCED_BLOCK_DRAW = [
    0,
    4,
    3,
    2,
    6,
    1,
    4,
    1,
    0,
    5,
    5,
    1,
    4,
    0,
    4,
    5,
    6,
    4,
    4,
    3,
    3,
    2,
    2,
    2,
    0,
    3,
]


class UniformValueNet:
    def __init__(self, action_size: int, value: float = 0.0) -> None:
        self.action_size = action_size
        self.value = value

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]:
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        return policy, self.value


class SolverOraclePlayer:
    def select_action(self, game: Game, state: State) -> int:
        _, optimal_moves = solve(state)
        return optimal_moves[0]


def _play(moves: list[int]):
    game = ConnectFour()
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return game, state


def _late_positions() -> tuple[ConnectFour, list[State]]:
    game, draw_state = _play(_DRAW_LINE)
    _, block_state = _play(_FORCED_BLOCK_DRAW)
    return game, [draw_state, block_state]


def test_connect_four_solver_anchor_metrics_are_bounded() -> None:
    game, positions = _late_positions()

    metrics = evaluate_connect_four_solver_anchor(
        UniformValueNet(game.action_size),
        game,
        positions=positions,
        n_positions=len(positions),
    )

    assert 0.0 <= metrics["eval/c4_value_mae"] <= 2.0
    assert 0.0 <= metrics["eval/c4_policy_match"] <= 1.0
    assert 0.0 <= metrics["eval/c4_blunder_rate"] <= 1.0
    assert metrics["eval/c4_solver_positions"] == len(positions)


def test_connect_four_solver_anchor_oracle_player_has_perfect_move_metrics() -> None:
    game, positions = _late_positions()

    metrics = evaluate_connect_four_solver_anchor(
        UniformValueNet(game.action_size),
        game,
        positions=positions,
        n_positions=len(positions),
        player=SolverOraclePlayer(),
    )

    if metrics["eval/c4_policy_match"] != 1.0:
        raise AssertionError(metrics)
    if metrics["eval/c4_blunder_rate"] != 0.0:
        raise AssertionError(metrics)


def test_train_agent_connect_four_eval_interval_includes_solver_anchor(
    monkeypatch,
) -> None:
    game = ConnectFour()
    anchor_metrics = {
        "eval/c4_value_mae": 0.5,
        "eval/c4_policy_match": 1.0,
        "eval/c4_blunder_rate": 0.0,
        "eval/c4_solver_positions": 2.0,
    }

    def fake_play_game(net, game_arg, cfg, temperature_schedule):
        policy = np.ones(game_arg.action_size, dtype=np.float32) / game_arg.action_size
        return [(game_arg.encode(game_arg.initial_state()), policy, 0.0)]

    def fake_compute_loss(*args, **kwargs):
        return torch.tensor(1.0), {}

    def fake_train_iteration(net, examples, **kwargs):
        return {"loss": 1.0}

    monkeypatch.setattr(arena, "play_game", fake_play_game)
    monkeypatch.setattr(arena, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(arena, "train_iteration", fake_train_iteration)
    monkeypatch.setattr(
        arena,
        "evaluate_ladder",
        lambda *args, **kwargs: {"eval/ladder_random_winrate": 0.0},
    )
    monkeypatch.setattr(
        arena,
        "evaluate_connect_four_solver_anchor",
        lambda *args, **kwargs: anchor_metrics,
    )

    _, metrics = train_agent(
        game,
        iterations=1,
        self_play_games_per_iteration=1,
        batch_size=1,
        epochs=1,
        gating_interval=99,
        eval_interval=1,
        seed=0,
    )

    for key, value in anchor_metrics.items():
        assert metrics[key] == value
