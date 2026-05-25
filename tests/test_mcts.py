"""Tests for the PUCT Monte Carlo Tree Search."""

from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS


class DummyNet:
    """Stand-in for AlphaZeroNet: fixed priors and value, no learning.

    With a uniform prior and value 0 the search is driven purely by the
    game's terminal outcomes, which isolates the PUCT/backup logic.
    """

    def __init__(self, action_size: int, priors=None, value: float = 0.0) -> None:
        self.action_size = action_size
        self._priors = priors
        self._value = value

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]:
        if self._priors is None:
            probs = np.ones(self.action_size, dtype=np.float32) / self.action_size
        else:
            probs = np.asarray(self._priors, dtype=np.float32)
        return probs, float(self._value)


@pytest.fixture
def game() -> TicTacToe:
    return TicTacToe()


def _state_after(game: TicTacToe, moves: list[int]):
    s = game.initial_state()
    for a in moves:
        s = game.apply_move(s, a)
    return s


# -- policy shape / normalization -------------------------------------------


def test_pi_is_distribution_over_legal_moves(game: TicTacToe) -> None:
    mcts = MCTS(DummyNet(game.action_size), game, num_simulations=64, seed=0)
    pi = mcts.run(game.initial_state())
    assert pi.shape == (game.action_size,)
    assert pi.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(pi >= 0)


def test_pi_zero_on_illegal_moves(game: TicTacToe) -> None:
    state = _state_after(game, [0, 3, 1, 4])  # occupied: 0,1,3,4
    mcts = MCTS(DummyNet(game.action_size), game, num_simulations=64, seed=0)
    pi = mcts.run(state)
    for illegal in (0, 1, 3, 4):
        assert pi[illegal] == 0
    legal_mass = sum(pi[a] for a in game.legal_moves(state))
    assert legal_mass == pytest.approx(1.0, abs=1e-9)


def test_run_on_terminal_state_returns_zero_policy(game: TicTacToe) -> None:
    terminal = _state_after(game, [0, 3, 1, 4, 2])  # X completes the top row
    assert game.is_terminal(terminal)
    mcts = MCTS(DummyNet(game.action_size), game, num_simulations=16, seed=0)
    pi = mcts.run(terminal)
    assert np.all(pi == 0)


# -- search quality: finds the forced win -----------------------------------


def test_finds_immediate_winning_move(game: TicTacToe) -> None:
    # X at 0,1 (top row, needs cell 2); O at 3,4. X to move and wins by playing 2.
    state = _state_after(game, [0, 3, 1, 4])
    assert game.current_player(state) == 1  # X to move
    mcts = MCTS(DummyNet(game.action_size), game, num_simulations=100, seed=0)
    pi = mcts.run(state)
    assert int(np.argmax(pi)) == 2  # the winning move dominates the visits
    assert pi[2] > 0.5


def test_blocks_opponents_immediate_win(game: TicTacToe) -> None:
    # O threatens to complete the left column (0,3 -> needs 6). It is X's move;
    # X has no immediate win, so the only non-losing reply is to block at 6.
    # X1 O0 X2 O3 X5 -> O to move; rebuild so X must block.
    # Sequence: O0, X1, O3, X5 would put O to move; instead set X to move facing
    # O's open column 0,3.
    state = _state_after(game, [1, 0, 5, 3])  # X:1,5  O:0,3 ; X to move
    assert game.current_player(state) == 1
    mcts = MCTS(DummyNet(game.action_size), game, num_simulations=200, seed=0)
    pi = mcts.run(state)
    assert int(np.argmax(pi)) == 6  # block O's column completion


# -- temperature sampling helper --------------------------------------------


def test_select_action_greedy_returns_argmax(game: TicTacToe) -> None:
    mcts = MCTS(DummyNet(game.action_size), game, seed=0)
    pi = np.zeros(game.action_size)
    pi[2] = 0.7
    pi[5] = 0.3
    assert mcts.select_action(pi, temperature=0.0) == 2


def test_select_action_only_samples_legal_moves(game: TicTacToe) -> None:
    mcts = MCTS(DummyNet(game.action_size), game, seed=0)
    pi = np.zeros(game.action_size)
    pi[2] = 0.5
    pi[7] = 0.5
    rng = np.random.default_rng(123)
    picks = {mcts.select_action(pi, temperature=1.0, rng=rng) for _ in range(200)}
    assert picks <= {2, 7}  # never selects a zero-probability action
    assert picks == {2, 7}  # both reachable under proportional sampling


def test_select_action_high_temperature_spreads_visits(game: TicTacToe) -> None:
    mcts = MCTS(DummyNet(game.action_size), game, seed=0)
    pi = np.zeros(game.action_size)
    pi[2] = 0.9
    pi[5] = 0.1
    rng = np.random.default_rng(7)
    counts = {2: 0, 5: 0}
    for _ in range(500):
        counts[mcts.select_action(pi, temperature=2.0, rng=rng)] += 1
    # Flattening pulls the minority move's share above its raw 10%.
    assert counts[5] > 0.1 * 500


# -- exploration noise -------------------------------------------------------


def test_root_noise_keeps_valid_policy_and_still_wins(game: TicTacToe) -> None:
    state = _state_after(game, [0, 3, 1, 4])  # X wins at 2
    mcts = MCTS(DummyNet(game.action_size), game, num_simulations=200, seed=1)
    pi = mcts.run(state, add_noise=True)
    assert pi.sum() == pytest.approx(1.0, abs=1e-9)
    for illegal in (0, 1, 3, 4):
        assert pi[illegal] == 0
    assert int(np.argmax(pi)) == 2
