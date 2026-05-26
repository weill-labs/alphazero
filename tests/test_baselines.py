"""Tests for simple search baselines."""

from __future__ import annotations

from collections.abc import Sequence

from alphazero.baselines import NegamaxPlayer
from alphazero.game import Game, State
from alphazero.games.connectfour import ConnectFour
from alphazero.games.tictactoe import TicTacToe


def _state_after_moves(game: Game, moves: Sequence[int]) -> State:
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return state


def test_negamax_takes_tictactoe_immediate_win() -> None:
    game = TicTacToe()
    state = _state_after_moves(game, [0, 3, 1, 4])

    action = NegamaxPlayer().select_action(game, state)

    assert action == 2
    assert game.winner(game.apply_move(state, action)) == game.current_player(state)


def test_negamax_blocks_tictactoe_immediate_loss() -> None:
    game = TicTacToe()
    state = _state_after_moves(game, [1, 0, 5, 3])

    action = NegamaxPlayer().select_action(game, state)

    assert action == 6
    next_state = game.apply_move(state, action)
    opponent_wins = [
        reply
        for reply in game.legal_moves(next_state)
        if game.winner(game.apply_move(next_state, reply))
        == game.current_player(next_state)
    ]
    assert opponent_wins == []


def test_negamax_takes_connect_four_immediate_win() -> None:
    game = ConnectFour()
    state = _state_after_moves(game, [0, 0, 1, 1, 2, 2])

    action = NegamaxPlayer().select_action(game, state)

    assert action == 3
    assert game.winner(game.apply_move(state, action)) == game.current_player(state)


def test_negamax_blocks_connect_four_immediate_loss() -> None:
    game = ConnectFour()
    state = _state_after_moves(game, [2, 0, 4, 0, 6, 0])

    action = NegamaxPlayer().select_action(game, state)

    assert action == 0
    next_state = game.apply_move(state, action)
    opponent_wins = [
        reply
        for reply in game.legal_moves(next_state)
        if game.winner(game.apply_move(next_state, reply))
        == game.current_player(next_state)
    ]
    assert opponent_wins == []
