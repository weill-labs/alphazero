"""Tests for AlphaZero self-play example generation."""

from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.selfplay import play_game


class DummyNet:
    """Fixed uniform policy/value network for deterministic self-play tests."""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]:
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        return policy, 0.0


def test_play_game_returns_examples_for_each_played_move() -> None:
    game = TicTacToe()
    examples = play_game(
        DummyNet(game.action_size),
        game,
        {"num_simulations": 16, "dirichlet_eps": 0.0, "seed": 0},
        temperature_schedule=0.0,
    )

    assert examples
    assert len(examples) <= game.action_size

    stone_counts: list[int] = []
    winner_candidates: set[int] = set()
    for move_index, (encoded_state, pi, z) in enumerate(examples):
        assert encoded_state.shape == (game.num_planes, *game.board_shape)
        assert encoded_state.dtype == np.float32
        assert pi.shape == (game.action_size,)
        assert pi.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(pi >= 0.0)
        assert z in {-1, 0, 1}

        stone_counts.append(int(encoded_state.sum()))
        if z != 0:
            mover = 1 if move_index % 2 == 0 else -1
            winner_candidates.add(mover * z)

    assert stone_counts == list(range(len(examples)))
    assert len(examples) == stone_counts[-1] + 1
    assert len(winner_candidates) <= 1
