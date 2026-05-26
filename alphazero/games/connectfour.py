"""Connect Four rules implementing the game-agnostic `Game` interface.

Board cells use an absolute encoding: +1 = first player ("X"), -1 = second
player ("O"), 0 = empty. Rows are stored top-to-bottom, so pieces fall toward
larger row indices.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from alphazero.game import Game

_ROWS = 6
_COLS = 7
_CONNECT = 4
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),  # horizontal
    (1, 0),  # vertical
    (1, 1),  # diagonal down-right
    (-1, 1),  # diagonal up-right
)


class ConnectFourState(NamedTuple):
    """Immutable, hashable Connect Four position.

    `board` holds absolute cell values (+1/-1/0) in top-to-bottom row order;
    `player` is whose turn it is (+1 or -1).
    """

    board: tuple[tuple[int, ...], ...]
    player: int


class ConnectFour(Game):
    action_size = _COLS
    board_shape = (_ROWS, _COLS)
    num_planes = 2

    def initial_state(self) -> ConnectFourState:
        return ConnectFourState(
            board=tuple(tuple(0 for _ in range(_COLS)) for _ in range(_ROWS)),
            player=1,
        )

    def current_player(self, s: ConnectFourState) -> int:
        return s.player

    def legal_moves(self, s: ConnectFourState) -> list[int]:
        if self.is_terminal(s):
            return []
        return [col for col in range(_COLS) if s.board[0][col] == 0]

    def apply_move(self, s: ConnectFourState, a: int) -> ConnectFourState:
        if not 0 <= a < self.action_size:
            raise ValueError(f"action {a} out of range [0, {self.action_size})")
        if self.is_terminal(s):
            raise ValueError("cannot move in a terminal state")
        if s.board[0][a] != 0:
            raise ValueError(f"column {a} is full")

        row_to_fill = next(
            row for row in range(_ROWS - 1, -1, -1) if s.board[row][a] == 0
        )
        new_board = [list(row) for row in s.board]
        new_board[row_to_fill][a] = s.player
        return ConnectFourState(
            board=tuple(tuple(row) for row in new_board),
            player=-s.player,
        )

    def is_terminal(self, s: ConnectFourState) -> bool:
        return self.winner(s) is not None

    def winner(self, s: ConnectFourState) -> int | None:
        for row in range(_ROWS):
            for col in range(_COLS):
                mark = s.board[row][col]
                if mark == 0:
                    continue
                for delta_row, delta_col in _DIRECTIONS:
                    if self._has_line(s, row, col, delta_row, delta_col, mark):
                        return mark

        if all(cell != 0 for row in s.board for cell in row):
            return 0
        return None

    def encode(self, s: ConnectFourState) -> np.ndarray:
        board = np.asarray(s.board, dtype=np.float32)
        current = (board == s.player).astype(np.float32)
        opponent = (board == -s.player).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    def __str__(self, s: ConnectFourState) -> str:
        glyph = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(glyph[cell] for cell in row) for row in s.board]
        return "\n".join([*rows, "0 1 2 3 4 5 6"])

    def _has_line(
        self,
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
