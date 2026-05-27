from __future__ import annotations

import numpy as np

from alphazero.c4_certify import certify_connect_four, sample_positions
from alphazero.c4_solver import solve
from alphazero.game import Game, State
from alphazero.games import ConnectFour


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


class ZeroNet:
    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]:
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        return policy, 0.0


class SolverOraclePlayer:
    def select_action(self, game: Game, state: State) -> int:
        _, optimal_moves = solve(state)
        return optimal_moves[0]


class FixedActionPlayer:
    def __init__(self, action: int) -> None:
        self.action = action

    def select_action(self, game: Game, state: State) -> int:
        if self.action not in game.legal_moves(state):
            raise ValueError(f"fixed action {self.action} is illegal")
        return self.action


def _play(moves: list[int]):
    game = ConnectFour()
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return game, state


def test_perfect_vs_solver_toy_case_scores_zero_blunders() -> None:
    game, state = _play(_FORCED_BLOCK_DRAW)

    report = certify_connect_four(
        ZeroNet(game.action_size),
        positions=[state],
        player=SolverOraclePlayer(),
        game=game,
    )

    assert report.evaluated_positions == 1
    assert np.isclose(report.policy_match_percent, 100.0)
    assert np.isclose(report.blunder_rate, 0.0)
    assert np.isclose(report.value_mae, 0.0)
    assert report.solved


def test_known_bad_move_is_flagged_as_blunder() -> None:
    game, state = _play(_FORCED_BLOCK_DRAW)

    report = certify_connect_four(
        ZeroNet(game.action_size),
        positions=[state],
        player=FixedActionPlayer(0),
        game=game,
    )

    assert report.evaluated_positions == 1
    assert np.isclose(report.policy_match_percent, 0.0)
    assert report.blunders == 1
    assert np.isclose(report.blunder_rate, 1.0)
    assert not report.solved


def test_position_sampling_is_deterministic() -> None:
    game = ConnectFour()

    first = sample_positions(game, sample_size=8, seed=123)
    second = sample_positions(game, sample_size=8, seed=123)

    assert first == second
    assert len(first) == 8
