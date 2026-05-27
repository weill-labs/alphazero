import random

import numpy as np
import pytest

from alphazero.games.connectfour import ConnectFour, ConnectFourState

_ROWS = 6
_COLS = 7
_CONNECT = 4
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
)


def _play(game: ConnectFour, moves: list[int]):
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return state


def _reference_winner(s: ConnectFourState) -> int | None:
    for row in range(_ROWS):
        for col in range(_COLS):
            mark = s.board[row][col]
            if mark == 0:
                continue
            for delta_row, delta_col in _DIRECTIONS:
                if _reference_has_line(s, row, col, delta_row, delta_col, mark):
                    return mark

    if all(cell != 0 for row in s.board for cell in row):
        return 0
    return None


def _reference_has_line(
    s: ConnectFourState,
    row: int,
    col: int,
    delta_row: int,
    delta_col: int,
    mark: int,
) -> bool:
    for offset in range(_CONNECT):
        next_row = row + offset * delta_row
        next_col = col + offset * delta_col
        if not (0 <= next_row < _ROWS and 0 <= next_col < _COLS):
            return False
        if s.board[next_row][next_col] != mark:
            return False
    return True


def _reference_apply_move(s: ConnectFourState, col: int) -> ConnectFourState:
    row_to_fill = next(
        row for row in range(_ROWS - 1, -1, -1) if s.board[row][col] == 0
    )
    new_board = [list(row) for row in s.board]
    new_board[row_to_fill][col] = s.player
    return ConnectFourState(
        board=tuple(tuple(row) for row in new_board),
        player=-s.player,
    )


def _reference_legal_moves(s: ConnectFourState) -> list[int]:
    if _reference_winner(s) is not None:
        return []
    return [col for col in range(_COLS) if s.board[0][col] == 0]


def test_horizontal_win():
    game = ConnectFour()

    state = _play(game, [0, 0, 1, 1, 2, 2, 3])

    assert game.winner(state) == 1
    assert game.is_terminal(state)


def test_vertical_win():
    game = ConnectFour()

    state = _play(game, [0, 1, 0, 1, 0, 1, 0])

    assert game.winner(state) == 1
    assert game.is_terminal(state)


def test_diagonal_up_right_win():
    game = ConnectFour()

    state = _play(game, [0, 1, 1, 2, 3, 2, 2, 3, 4, 3, 3])

    assert game.winner(state) == 1
    assert game.is_terminal(state)


def test_diagonal_down_right_win():
    game = ConnectFour()

    state = _play(game, [6, 5, 5, 4, 3, 4, 4, 3, 2, 3, 3])

    assert game.winner(state) == 1
    assert game.is_terminal(state)


def test_winner_matches_full_scan_reference_for_random_positions():
    game = ConnectFour()
    rng = random.Random(20260527)
    states = [
        game.initial_state(),
        ConnectFourState(
            board=(
                (-1, -1, -1, -1, 0, 0, 0),
                (1, 1, 1, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
            ),
            player=1,
        ),
    ]

    for _ in range(2_000):
        board = tuple(
            tuple(rng.choice((-1, 0, 1)) for _ in range(_COLS)) for _ in range(_ROWS)
        )
        states.append(ConnectFourState(board=board, player=rng.choice((-1, 1))))

    terminal_count = 0
    for _ in range(250):
        state = game.initial_state()
        states.append(state)

        while _reference_winner(state) is None:
            legal_moves = _reference_legal_moves(state)
            if not legal_moves:
                break
            state = _reference_apply_move(state, rng.choice(legal_moves))
            states.append(state)

        if _reference_winner(state) is not None:
            terminal_count += 1

    if terminal_count != 250:
        pytest.fail(f"expected 250 terminal games, got {terminal_count}")

    for state in states:
        expected = _reference_winner(state)
        actual = game.winner(state)
        if actual != expected:
            pytest.fail(f"winner mismatch: expected {expected}, got {actual}")


def test_full_board_draw_has_no_winner():
    game = ConnectFour()
    moves = [
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
        6,
        0,
        1,
        1,
    ]

    state = _play(game, moves)

    assert game.winner(state) == 0
    assert game.is_terminal(state)
    assert game.legal_moves(state) == []


def test_full_column_is_excluded_from_legal_moves():
    game = ConnectFour()

    state = _play(game, [0, 0, 0, 0, 0, 0])

    assert game.winner(state) is None
    assert 0 not in game.legal_moves(state)
    assert game.legal_moves(state) == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="column 0 is full"):
        game.apply_move(state, 0)


def test_apply_move_does_not_mutate_input():
    game = ConnectFour()
    state = game.initial_state()

    next_state = game.apply_move(state, 3)

    assert isinstance(hash(state), int)
    assert state.board == tuple(tuple(0 for _ in range(7)) for _ in range(6))
    assert state.player == 1
    assert next_state.board[5][3] == 1
    assert next_state.player == -1
    assert next_state is not state


def test_encode_is_canonical_to_player_to_move():
    game = ConnectFour()

    after_x = game.apply_move(game.initial_state(), 3)
    encoded_after_x = game.encode(after_x)

    assert encoded_after_x.shape == (2, 6, 7)
    assert encoded_after_x.dtype == np.float32
    assert encoded_after_x[0].sum() == 0
    assert encoded_after_x[1, 5, 3] == 1

    after_o = game.apply_move(after_x, 4)
    encoded_after_o = game.encode(after_o)

    assert encoded_after_o[0, 5, 3] == 1
    assert encoded_after_o[1, 5, 4] == 1
