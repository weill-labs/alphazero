"""Tests for the Game interface and the tic-tac-toe implementation."""

from __future__ import annotations

import numpy as np
import pytest

from alphazero.game import Game
from alphazero.games.tictactoe import TicTacToe, TicTacToeState


@pytest.fixture
def g() -> TicTacToe:
    return TicTacToe()


def _play(g: TicTacToe, moves: list[int]) -> TicTacToeState:
    """Apply a sequence of moves from the initial state."""
    s = g.initial_state()
    for a in moves:
        s = g.apply_move(s, a)
    return s


# --- abstract base contract -------------------------------------------------


def test_game_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        Game()  # type: ignore[abstract]


def test_tictactoe_exposes_contract_attributes(g: TicTacToe) -> None:
    assert isinstance(g, Game)
    assert g.action_size == 9
    assert g.board_shape == (3, 3)
    assert g.num_planes == 2


# --- initial state ----------------------------------------------------------


def test_initial_state(g: TicTacToe) -> None:
    s = g.initial_state()
    assert g.current_player(s) == 1
    assert sorted(g.legal_moves(s)) == list(range(9))
    assert not g.is_terminal(s)
    assert g.winner(s) is None


# --- moves ------------------------------------------------------------------


def test_apply_move_does_not_mutate_original(g: TicTacToe) -> None:
    s = g.initial_state()
    s2 = g.apply_move(s, 4)
    assert s.board == (0,) * 9  # original untouched
    assert s2.board[4] == 1
    assert g.current_player(s2) == -1  # turn passed to the other player


def test_action_index_is_row_times_three_plus_col(g: TicTacToe) -> None:
    s = g.apply_move(g.initial_state(), 1 * 3 + 2)  # (row=1, col=2)
    assert s.board[5] == 1


def test_legal_moves_excludes_occupied_cells(g: TicTacToe) -> None:
    s = _play(g, [4, 0])
    assert 4 not in g.legal_moves(s)
    assert 0 not in g.legal_moves(s)
    assert len(g.legal_moves(s)) == 7


@pytest.mark.parametrize("bad_move", [-1, 9, 100])
def test_apply_move_rejects_out_of_range(g: TicTacToe, bad_move: int) -> None:
    with pytest.raises(ValueError):
        g.apply_move(g.initial_state(), bad_move)


def test_apply_move_rejects_occupied_cell(g: TicTacToe) -> None:
    s = g.apply_move(g.initial_state(), 0)
    with pytest.raises(ValueError):
        g.apply_move(s, 0)


def test_apply_move_rejects_terminal_state(g: TicTacToe) -> None:
    # X wins on the top row: X0 O3 X1 O4 X2
    s = _play(g, [0, 3, 1, 4, 2])
    assert g.is_terminal(s)
    with pytest.raises(ValueError):
        g.apply_move(s, 5)


# --- terminal detection -----------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ],
)
def test_each_winning_line_for_player_x(
    g: TicTacToe, line: tuple[int, int, int]
) -> None:
    others = [i for i in range(9) if i not in line]
    # Interleave X's winning moves with harmless O moves.
    seq = [line[0], others[0], line[1], others[1], line[2]]
    s = _play(g, seq)
    assert g.winner(s) == 1
    assert g.is_terminal(s)
    assert g.legal_moves(s) == []


def test_player_o_can_win(g: TicTacToe) -> None:
    # O takes the left column (0,3,6); X plays elsewhere and does not complete a line.
    # X1 O0 X2 O3 X5 O6
    s = _play(g, [1, 0, 2, 3, 5, 6])
    assert g.winner(s) == -1
    assert g.is_terminal(s)


def test_draw_is_full_board_with_no_line(g: TicTacToe) -> None:
    # X O X / X O O / O X X  -> reachable full board, no three-in-a-row.
    draw_board = (1, -1, 1, 1, -1, -1, -1, 1, 1)
    s = TicTacToeState(board=draw_board, player=1)
    assert g.winner(s) == 0
    assert g.is_terminal(s)
    assert g.legal_moves(s) == []


# --- encoding (canonical perspective) ---------------------------------------


def test_encode_shape_and_dtype(g: TicTacToe) -> None:
    enc = g.encode(g.initial_state())
    assert enc.shape == (2, 3, 3)
    assert enc.dtype == np.float32
    # Planes are one-hot, so cell counts are exact integers.
    assert int(enc.sum()) == 0  # empty board has no pieces on either plane


def test_encode_is_from_player_to_move_perspective(g: TicTacToe) -> None:
    # X plays center -> now O (-1) is to move.
    s = g.apply_move(g.initial_state(), 4)
    enc = g.encode(s)
    # Plane 0 = current player's (O's) pieces: none yet.
    assert int(enc[0].sum()) == 0
    # Plane 1 = opponent's (X's) pieces: the center.
    assert int(enc[1, 1, 1]) == 1
    assert int(enc[1].sum()) == 1


def test_encode_perspective_flips_each_ply(g: TicTacToe) -> None:
    # X center, then O corner -> now X is to move again.
    s = _play(g, [4, 0])
    enc = g.encode(s)
    # Current player is X: own piece (center) on plane 0, O's corner on plane 1.
    assert int(enc[0, 1, 1]) == 1
    assert int(enc[1, 0, 0]) == 1
    assert int(enc[0].sum()) == 1
    assert int(enc[1].sum()) == 1


# --- state value semantics --------------------------------------------------


def test_states_are_hashable_and_value_equal(g: TicTacToe) -> None:
    table = {g.initial_state(): "start"}
    assert table[g.initial_state()] == "start"  # equal states collide as keys
    s1 = g.apply_move(g.initial_state(), 0)
    s2 = g.apply_move(g.initial_state(), 0)
    assert s1 == s2
    assert hash(s1) == hash(s2)


def test_str_renders_three_rows_with_marks(g: TicTacToe) -> None:
    s = _play(g, [0, 4])  # X at 0, O at center
    rendered = g.__str__(s)
    lines = rendered.splitlines()
    assert len(lines) == 3
    assert "X" in rendered and "O" in rendered
    assert lines[0].startswith("X")
