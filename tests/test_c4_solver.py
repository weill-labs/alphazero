from __future__ import annotations

import pytest

from alphazero.c4_solver import NodeBudgetExceeded, solve
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
    6,
    0,
    1,
    1,
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

_TEST_MAX_NODES = 250_000


def _play(moves: list[int]):
    game = ConnectFour()
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return game, state


def test_opening_book_certifies_empty_board_and_center_reply() -> None:
    game = ConnectFour()
    state = game.initial_state()

    assert solve(state, max_nodes=1) == (1, [3])

    center_reply_state = game.apply_move(state, 3)
    assert solve(center_reply_state, max_nodes=1) == (
        -1,
        game.legal_moves(center_reply_state),
    )


def test_midgame_position_still_uses_live_search() -> None:
    _, state = _play(_FORCED_BLOCK_DRAW)

    with pytest.raises(NodeBudgetExceeded):
        solve(state, max_nodes=1)

    assert solve(state, max_nodes=_TEST_MAX_NODES) == (0, [1])


def test_late_immediate_winning_move_is_optimal() -> None:
    game, state = _play(_DRAW_LINE[:32])

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)

    assert game.winner(game.apply_move(state, 0)) == state.player
    assert value == 1
    assert 0 in optimal_moves


def test_forced_single_non_losing_move_blocks_direct_threat() -> None:
    game, state = _play(_FORCED_BLOCK_DRAW)

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)

    assert value == 0
    assert optimal_moves == [1]
    for move in [0, 2, 3, 5, 6]:
        reply_state = game.apply_move(state, move)
        assert game.winner(game.apply_move(reply_state, 1)) == state.player * -1


def test_near_full_board_one_move_from_draw() -> None:
    game, state = _play(_DRAW_LINE[:41])

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)
    final_state = game.apply_move(state, optimal_moves[0])

    assert value == 0
    assert optimal_moves == [1]
    assert game.winner(final_state) == 0


def test_value_is_negated_after_optimal_move() -> None:
    game, state = _play(_FORCED_BLOCK_DRAW)

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)
    next_value, _ = solve(
        game.apply_move(state, optimal_moves[0]),
        max_nodes=_TEST_MAX_NODES,
    )

    assert value == -next_value


def test_multiple_optimal_moves_are_all_returned() -> None:
    game, state = _play(_DRAW_LINE[:38])

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)
    expected_moves = []
    for move in game.legal_moves(state):
        child_value, _ = solve(
            game.apply_move(state, move),
            max_nodes=_TEST_MAX_NODES,
        )
        if -child_value == value:
            expected_moves.append(move)

    assert len(expected_moves) > 1
    assert optimal_moves == expected_moves


def test_hand_verified_shallow_endgame_win() -> None:
    _, state = _play(_DRAW_LINE[:36])

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)

    assert value == 1
    assert optimal_moves == [0]


def test_hand_verified_shallow_endgame_draw() -> None:
    game, state = _play(_DRAW_LINE[:38])

    value, optimal_moves = solve(state, max_nodes=_TEST_MAX_NODES)

    assert value == 0
    assert optimal_moves == game.legal_moves(state)


def test_node_budget_guard_raises_before_hanging() -> None:
    _, state = _play(_DRAW_LINE[:36])

    with pytest.raises(NodeBudgetExceeded):
        solve(state, max_nodes=1)
