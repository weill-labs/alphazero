"""Game registry.

Maps game names to their `Game` implementations so that adding a new game is a
single edit here, rather than touching every CLI/entrypoint that constructs a
game by name.
"""

from __future__ import annotations

from collections.abc import Callable

from alphazero.game import Game
from alphazero.games.connectfour import ConnectFour
from alphazero.games.go import Go
from alphazero.games.gomoku import Gomoku
from alphazero.games.tictactoe import TicTacToe

GAMES: dict[str, Callable[[], Game]] = {
    "tictactoe": TicTacToe,
    "connectfour": ConnectFour,
    "gomoku": Gomoku,
    "go": Go,
}

GAME_CHOICES: tuple[str, ...] = tuple(GAMES)


def game_from_name(name: str) -> Game:
    try:
        factory = GAMES[name]
    except KeyError:
        raise ValueError(f"unknown game {name!r}; choose from {GAME_CHOICES}") from None
    return factory()
