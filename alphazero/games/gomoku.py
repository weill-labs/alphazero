"""Gomoku (five-in-a-row) rules implementing the game-agnostic `Game` interface.

Unlike Connect Four there is no gravity: a move places the current player's
mark on any empty cell. Cells use an absolute encoding (+1 = first player "X",
-1 = second player "O", 0 = empty) and actions index cells in row-major order,
so action ``a`` maps to ``(a // size, a % size)``.

Board size and the win length are constructor arguments (default 9x9, connect
5) so the same rules cover the standard small board and tiny test boards.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from alphazero.game import Game

_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),  # horizontal
    (1, 0),  # vertical
    (1, 1),  # diagonal down-right
    (-1, 1),  # diagonal up-right
)


class GomokuState(NamedTuple):
    """Immutable, hashable Gomoku position.

    `board` holds absolute cell values (+1/-1/0) in row-major order; `player`
    is whose turn it is (+1 or -1).
    """

    board: tuple[tuple[int, ...], ...]
    player: int


class Gomoku(Game):
    num_planes = 2

    def __init__(self, *, size: int = 9, connect: int = 5) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if connect <= 0:
            raise ValueError("connect must be positive")
        self.size = size
        self.connect = connect
        self.action_size = size * size
        self.board_shape = (size, size)

    def initial_state(self) -> GomokuState:
        return GomokuState(
            board=tuple(tuple(0 for _ in range(self.size)) for _ in range(self.size)),
            player=1,
        )

    def current_player(self, s: GomokuState) -> int:
        return s.player

    def legal_moves(self, s: GomokuState) -> list[int]:
        if self.is_terminal(s):
            return []
        return [
            row * self.size + col
            for row in range(self.size)
            for col in range(self.size)
            if s.board[row][col] == 0
        ]

    def apply_move(self, s: GomokuState, a: int) -> GomokuState:
        if not 0 <= a < self.action_size:
            raise ValueError(f"action {a} out of range [0, {self.action_size})")
        if self.is_terminal(s):
            raise ValueError("cannot move in a terminal state")
        row, col = divmod(a, self.size)
        if s.board[row][col] != 0:
            raise ValueError(f"cell {a} ({row}, {col}) is occupied")

        new_board = [list(board_row) for board_row in s.board]
        new_board[row][col] = s.player
        return GomokuState(
            board=tuple(tuple(board_row) for board_row in new_board),
            player=-s.player,
        )

    def is_terminal(self, s: GomokuState) -> bool:
        return self.winner(s) is not None

    def winner(self, s: GomokuState) -> int | None:
        for row in range(self.size):
            for col in range(self.size):
                mark = s.board[row][col]
                if mark == 0:
                    continue
                for delta_row, delta_col in _DIRECTIONS:
                    if self._has_line(s, row, col, delta_row, delta_col, mark):
                        return mark

        if all(cell != 0 for board_row in s.board for cell in board_row):
            return 0
        return None

    def encode(self, s: GomokuState) -> np.ndarray:
        board = np.asarray(s.board, dtype=np.float32)
        current = (board == s.player).astype(np.float32)
        opponent = (board == -s.player).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    def __str__(self, s: GomokuState) -> str:
        glyph = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(glyph[cell] for cell in board_row) for board_row in s.board]
        header = " ".join(str(col % 10) for col in range(self.size))
        return "\n".join([*rows, header])

    def _has_line(
        self,
        s: GomokuState,
        row: int,
        col: int,
        delta_row: int,
        delta_col: int,
        mark: int,
    ) -> bool:
        for offset in range(self.connect):
            next_row = row + offset * delta_row
            next_col = col + offset * delta_col
            if not (0 <= next_row < self.size and 0 <= next_col < self.size):
                return False
            if s.board[next_row][next_col] != mark:
                return False
        return True
