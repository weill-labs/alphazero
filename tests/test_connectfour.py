import numpy as np
import pytest

from alphazero.games.connectfour import ConnectFour


def _play(game: ConnectFour, moves: list[int]):
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return state


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
