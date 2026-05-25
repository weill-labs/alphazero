"""Tic-tac-toe rules implementing the game-agnostic `Game` interface.

Board cells use an *absolute* encoding: +1 = first player ("X"), -1 = second
player ("O"), 0 = empty. Cell index ``i`` maps to ``(row, col) = (i // 3, i % 3)``,
so ``action = row * 3 + col``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from alphazero.game import Game

# The eight ways to make three-in-a-row, as cell-index triples.
_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # columns
    (0, 4, 8),
    (2, 4, 6),  # diagonals
)


class TicTacToeState(NamedTuple):
    """Immutable, hashable tic-tac-toe position.

    `board` holds 9 cells in absolute terms (+1/-1/0); `player` is whose turn
    it is (+1 or -1). Keeping `player` explicit avoids re-deriving it from move
    parity and keeps the `Game.current_player` contract trivial.
    """

    board: tuple[int, ...]  # length 9, values in {+1, -1, 0}
    player: int  # +1 or -1: the player to move


class TicTacToe(Game):
    action_size = 9
    board_shape = (3, 3)
    num_planes = 2

    def initial_state(self) -> TicTacToeState:
        return TicTacToeState(board=(0,) * 9, player=1)

    def current_player(self, s: TicTacToeState) -> int:
        return s.player

    def winner(self, s: TicTacToeState) -> int | None:
        for a, b, c in _LINES:
            mark = s.board[a]
            if mark != 0 and mark == s.board[b] == s.board[c]:
                return mark
        if all(cell != 0 for cell in s.board):
            return 0  # full board, no line: draw
        return None

    def is_terminal(self, s: TicTacToeState) -> bool:
        return self.winner(s) is not None

    def legal_moves(self, s: TicTacToeState) -> list[int]:
        if self.is_terminal(s):
            return []
        return [i for i, cell in enumerate(s.board) if cell == 0]

    def apply_move(self, s: TicTacToeState, a: int) -> TicTacToeState:
        if not 0 <= a < self.action_size:
            raise ValueError(f"action {a} out of range [0, {self.action_size})")
        if s.board[a] != 0:
            raise ValueError(f"cell {a} is already occupied in {s.board}")
        if self.is_terminal(s):
            raise ValueError("cannot move in a terminal state")
        new_board = list(s.board)
        new_board[a] = s.player
        return TicTacToeState(board=tuple(new_board), player=-s.player)

    def encode(self, s: TicTacToeState) -> np.ndarray:
        board = np.asarray(s.board, dtype=np.float32).reshape(self.board_shape)
        current = (board == s.player).astype(np.float32)
        opponent = (board == -s.player).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    def __str__(self, s: TicTacToeState) -> str:
        glyph = {1: "X", -1: "O", 0: "."}
        return "\n".join(
            " ".join(glyph[s.board[row * 3 + col]] for col in range(3))
            for row in range(3)
        )
