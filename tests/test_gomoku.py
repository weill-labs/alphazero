"""Tests for the Gomoku (five-in-a-row) rules."""

from __future__ import annotations

import pytest

from alphazero.games import GAME_CHOICES, game_from_name
from alphazero.games.gomoku import Gomoku, GomokuState


def _state(
    size: int, cells: dict[tuple[int, int], int], player: int = 1
) -> GomokuState:
    board = [[0] * size for _ in range(size)]
    for (row, col), mark in cells.items():
        board[row][col] = mark
    return GomokuState(board=tuple(tuple(row) for row in board), player=player)


def test_initial_state_is_empty_with_first_player_to_move() -> None:
    game = Gomoku()
    state = game.initial_state()

    assert game.action_size == 81
    assert game.board_shape == (9, 9)
    assert game.num_planes == 2
    assert state.player == 1
    assert all(cell == 0 for row in state.board for cell in row)
    assert game.legal_moves(state) == list(range(81))


def test_apply_move_places_mark_and_switches_player() -> None:
    game = Gomoku()
    state = game.apply_move(game.initial_state(), 40)  # center: (4, 4)

    assert state.board[4][4] == 1
    assert state.player == -1
    assert 40 not in game.legal_moves(state)


@pytest.mark.parametrize(
    "bad_action",
    [-1, 81, 100],
)
def test_apply_move_rejects_out_of_range_action(bad_action: int) -> None:
    game = Gomoku()
    with pytest.raises(ValueError, match="out of range"):
        game.apply_move(game.initial_state(), bad_action)


def test_apply_move_rejects_occupied_cell() -> None:
    game = Gomoku()
    state = game.apply_move(game.initial_state(), 40)
    with pytest.raises(ValueError, match="occupied"):
        game.apply_move(state, 40)


def test_apply_move_rejects_terminal_state() -> None:
    game = Gomoku()
    won = _state(9, {(0, col): 1 for col in range(5)})
    with pytest.raises(ValueError, match="terminal"):
        game.apply_move(won, 40)


@pytest.mark.parametrize(
    "cells",
    [
        {(0, col): 1 for col in range(5)},  # horizontal
        {(row, 0): 1 for row in range(5)},  # vertical
        {(i, i): 1 for i in range(5)},  # diagonal down-right
        {(4 - i, i): 1 for i in range(5)},  # diagonal up-right
    ],
)
def test_five_in_a_row_wins(cells: dict[tuple[int, int], int]) -> None:
    game = Gomoku()
    assert game.winner(_state(9, cells)) == 1
    assert game.is_terminal(_state(9, cells))


def test_winner_reports_second_player() -> None:
    game = Gomoku()
    assert game.winner(_state(9, {(2, col): -1 for col in range(5)})) == -1


def test_four_in_a_row_is_not_a_win() -> None:
    game = Gomoku()
    state = _state(9, {(0, col): 1 for col in range(4)})
    assert game.winner(state) is None
    assert not game.is_terminal(state)


def test_full_board_without_a_line_is_a_draw() -> None:
    # A 3x3 board can never hold five in a row, so a full board is a draw.
    game = Gomoku(size=3, connect=5)
    full = _state(
        3, {(r, c): (1 if (r + c) % 2 == 0 else -1) for r in range(3) for c in range(3)}
    )

    assert game.winner(full) == 0
    assert game.is_terminal(full)
    assert game.legal_moves(full) == []


def test_encode_is_from_current_player_perspective() -> None:
    game = Gomoku()
    cells = {(0, 0): 1, (0, 1): -1}

    encoded_p1 = game.encode(_state(9, cells, player=1))
    assert encoded_p1.shape == (2, 9, 9)
    assert encoded_p1[0, 0, 0] == 1.0  # current (+1) on plane 0
    assert encoded_p1[1, 0, 1] == 1.0  # opponent (-1) on plane 1

    encoded_p2 = game.encode(_state(9, cells, player=-1))
    assert encoded_p2[0, 0, 1] == 1.0  # current (-1) now on plane 0
    assert encoded_p2[1, 0, 0] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 0}, "size must be positive"),
        ({"connect": 0}, "connect must be positive"),
    ],
)
def test_constructor_rejects_invalid_config(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Gomoku(**kwargs)


def test_registered_in_game_registry() -> None:
    assert "gomoku" in GAME_CHOICES
    game = game_from_name("gomoku")
    assert isinstance(game, Gomoku)
    assert game.encode(game.initial_state()).shape == (
        game.num_planes,
        *game.board_shape,
    )
