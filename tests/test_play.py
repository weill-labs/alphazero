"""Tests for the human-vs-agent play CLI helpers."""

from __future__ import annotations

import pytest

from alphazero.game import Game, State
from alphazero.games.tictactoe import TicTacToe
from alphazero.play import (
    outcome_message,
    parse_legal_move,
    parse_move,
    play_human_vs_agent,
    validate_move,
)


class ScriptedPlayer:
    def __init__(self, moves: list[int]) -> None:
        self._moves = iter(moves)

    def select_action(self, game: Game, state: State) -> int:
        return next(self._moves)


def test_parse_move_accepts_integer_text() -> None:
    assert parse_move("4") == 4
    assert parse_move("  6\n") == 6


def test_parse_move_rejects_non_integer_text() -> None:
    with pytest.raises(ValueError, match="integer"):
        parse_move("center")


def test_validate_move_rejects_occupied_cell() -> None:
    game = TicTacToe()
    state = game.apply_move(game.initial_state(), 0)

    with pytest.raises(ValueError, match="not legal"):
        validate_move(game, state, 0)


def test_parse_legal_move_combines_parsing_and_validation() -> None:
    game = TicTacToe()
    state = game.initial_state()

    assert parse_legal_move("8", game, state) == 8

    with pytest.raises(ValueError, match="not legal"):
        parse_legal_move("9", game, state)


def test_scripted_human_first_game_reports_human_win() -> None:
    game = TicTacToe()
    human_moves = iter(["0", "1", "2"])
    output: list[str] = []

    result = play_human_vs_agent(
        game,
        ScriptedPlayer([3, 4]),
        human_first=True,
        input_fn=lambda prompt: next(human_moves),
        output_fn=output.append,
    )

    assert result.winner == 1
    assert result.human_player == 1
    assert result.message == "You win."
    assert output[-1] == "You win."
    assert "Agent plays 3." in output
    assert any("X X X" in board for board in output)


def test_scripted_loop_reprompts_after_invalid_human_input() -> None:
    game = TicTacToe()
    human_moves = iter(["nope", "9", "0", "1", "2"])
    output: list[str] = []

    result = play_human_vs_agent(
        game,
        ScriptedPlayer([3, 4]),
        human_first=True,
        input_fn=lambda prompt: next(human_moves),
        output_fn=output.append,
    )

    assert result.message == "You win."
    assert any("Invalid move: enter an integer move" == line for line in output)
    assert any("Invalid move: move 9 is not legal" in line for line in output)


def test_outcome_message_labels_human_result() -> None:
    assert outcome_message(1, 1) == "You win."
    assert outcome_message(-1, 1) == "You lose."
    assert outcome_message(0, 1) == "Draw."
