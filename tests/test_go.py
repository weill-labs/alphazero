"""Tests for the Go rules (captures, suicide, ko, scoring)."""

from __future__ import annotations

import pytest

from alphazero.games import GAME_CHOICES, game_from_name
from alphazero.games.go import Go, GoState


def _state(
    rows: list[list[int]],
    *,
    player: int = 1,
    ko_point: int | None = None,
    passes: int = 0,
) -> GoState:
    return GoState(
        board=tuple(tuple(row) for row in rows),
        player=player,
        ko_point=ko_point,
        passes=passes,
    )


def test_initial_state_and_action_space() -> None:
    game = Go()  # 7x7
    state = game.initial_state()

    assert game.action_size == 50  # 49 points + pass
    assert game.pass_action == 49
    assert game.board_shape == (7, 7)
    assert state.player == 1
    # Every point plus pass is legal on the empty board.
    assert len(game.legal_moves(state)) == 50
    assert game.pass_action in game.legal_moves(state)


def test_two_consecutive_passes_end_the_game() -> None:
    game = Go(size=3)
    state = game.initial_state()

    state = game.apply_move(state, game.pass_action)
    assert state.passes == 1 and not game.is_terminal(state)

    state = game.apply_move(state, game.pass_action)
    assert state.passes == 2
    assert game.is_terminal(state)
    assert game.legal_moves(state) == []


def test_placing_a_stone_captures_a_surrounded_group() -> None:
    game = Go(size=3)
    # White (0,1) has its last liberty at (0,0); Black plays it.
    state = _state(
        [
            [0, -1, 1],
            [-1, 1, 0],
            [0, 0, 0],
        ]
    )
    result = game.apply_move(state, 0)  # Black plays (0, 0)

    assert result.board[0][0] == 1
    assert result.board[0][1] == 0  # captured white stone removed
    assert result.player == -1


def test_plain_suicide_is_illegal() -> None:
    game = Go(size=3)
    # (0,0) is bordered only by white groups that keep their own liberties.
    state = _state(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, 0],
        ]
    )

    assert 0 not in game.legal_moves(state)
    with pytest.raises(ValueError, match="suicide"):
        game.apply_move(state, 0)


def test_suicide_that_captures_is_legal() -> None:
    game = Go(size=3)
    # Playing (0,0) fills white (0,1)'s last liberty, capturing it and giving
    # the new black stone a liberty — so it is legal despite looking like suicide.
    state = _state(
        [
            [0, -1, 1],
            [-1, 1, 0],
            [0, 0, 0],
        ]
    )

    assert 0 in game.legal_moves(state)
    result = game.apply_move(state, 0)
    assert result.board[0][0] == 1
    assert result.board[0][1] == 0


def test_simple_ko_forbids_immediate_recapture() -> None:
    game = Go(size=4)
    # Black to play (1,2)=index 6 captures the lone white stone at (1,1)=index 5
    # and ends in atari, creating a ko.
    state = _state(
        [
            [0, 1, -1, 0],
            [1, -1, 0, -1],
            [0, 1, -1, 0],
            [0, 0, 0, 0],
        ]
    )
    result = game.apply_move(state, 6)

    assert result.board[1][1] == 0  # white captured
    assert result.board[1][2] == 1
    assert result.ko_point == 5  # the captured point
    # White (now to move) may not immediately recapture at the ko point.
    assert 5 not in game.legal_moves(result)
    with pytest.raises(ValueError, match="ko"):
        game.apply_move(result, 5)


def test_passing_clears_the_ko_ban() -> None:
    game = Go(size=4)
    state = _state(
        [
            [0, 1, -1, 0],
            [1, -1, 0, -1],
            [0, 1, -1, 0],
            [0, 0, 0, 0],
        ]
    )
    after_capture = game.apply_move(state, 6)
    after_pass = game.apply_move(after_capture, game.pass_action)
    assert after_pass.ko_point is None


def test_empty_board_scores_for_white_by_komi() -> None:
    game = Go(size=3, komi=7.5)
    # Two passes on an empty board: no territory belongs to anyone, so komi decides.
    terminal = _state([[0, 0, 0]] * 3, passes=2)
    assert game.winner(terminal) == -1


def test_area_scoring_attributes_surrounded_territory() -> None:
    game = Go(size=3, komi=7.5)
    # Black rings the board; the single empty center is black territory.
    terminal = _state(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        passes=2,
    )
    # 8 stones + 1 territory = 9 for black, 0 for white; 9 - 7.5 > 0.
    assert game.winner(terminal) == 1


def test_winner_is_none_before_two_passes() -> None:
    game = Go(size=3)
    assert game.winner(game.initial_state()) is None


def test_constructor_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        Go(size=0)


def test_registered_in_game_registry() -> None:
    assert "go" in GAME_CHOICES
    game = game_from_name("go")
    assert isinstance(game, Go)
    assert game.encode(game.initial_state()).shape == (
        game.num_planes,
        *game.board_shape,
    )
