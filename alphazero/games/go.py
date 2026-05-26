"""Go rules implementing the game-agnostic `Game` interface.

Scope and rule choices (documented because Go has several variants):

- **Board / actions:** square board of side ``size`` (default 7). Actions index
  points in row-major order; the extra action ``size*size`` is *pass*, so
  ``action_size == size*size + 1``.
- **Captures / suicide:** placing a stone first removes opponent groups left
  with no liberties, then the moving group is checked — a move that leaves its
  own group without liberties (suicide) is illegal.
- **Ko:** simple single-point ko (a move may not immediately recreate the prior
  position by recapturing in a one-stone ko). This is cheaper than full
  positional superko, which would require carrying the whole position history
  in every state; it covers the common case.
- **End / scoring:** two consecutive passes end the game. Scoring is
  Tromp-Taylor area scoring (stones on the board plus empty regions bordered by
  a single color) with ``komi`` for White (-1). A half-integer komi (default
  7.5) keeps every game decisive (no draws).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from alphazero.game import Game


class GoState(NamedTuple):
    """Immutable, hashable Go position.

    `board` holds absolute point values (+1/-1/0) in row-major order; `player`
    is to move; `ko_point` is the index forbidden by simple ko (or None);
    `passes` counts consecutive passes (two ends the game).
    """

    board: tuple[tuple[int, ...], ...]
    player: int
    ko_point: int | None
    passes: int


class Go(Game):
    num_planes = 2

    def __init__(self, *, size: int = 7, komi: float = 7.5) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.komi = komi
        self.pass_action = size * size
        self.action_size = size * size + 1
        self.board_shape = (size, size)

    def initial_state(self) -> GoState:
        board = tuple(tuple(0 for _ in range(self.size)) for _ in range(self.size))
        return GoState(board=board, player=1, ko_point=None, passes=0)

    def current_player(self, s: GoState) -> int:
        return s.player

    def legal_moves(self, s: GoState) -> list[int]:
        if self.is_terminal(s):
            return []
        moves = [
            point
            for point in range(self.size * self.size)
            if self._is_legal_point(s, point)
        ]
        moves.append(self.pass_action)  # passing is always legal
        return moves

    def apply_move(self, s: GoState, a: int) -> GoState:
        if not 0 <= a < self.action_size:
            raise ValueError(f"action {a} out of range [0, {self.action_size})")
        if self.is_terminal(s):
            raise ValueError("cannot move in a terminal state")

        if a == self.pass_action:
            return GoState(
                board=s.board,
                player=-s.player,
                ko_point=None,
                passes=s.passes + 1,
            )

        if s.board[a // self.size][a % self.size] != 0:
            raise ValueError(f"point {a} is occupied")
        if a == s.ko_point:
            raise ValueError(f"point {a} is forbidden by ko")

        new_board, captured = self._place(s.board, a, s.player)
        group, liberties = self._group(new_board, a)
        if not liberties:
            raise ValueError(f"point {a} is suicide")

        # Simple ko: a single-stone capture that leaves the mover in atari is
        # the only position the opponent could instantly recreate.
        ko_point = (
            captured[0]
            if len(captured) == 1 and len(group) == 1 and len(liberties) == 1
            else None
        )
        return GoState(
            board=tuple(tuple(row) for row in new_board),
            player=-s.player,
            ko_point=ko_point,
            passes=0,
        )

    def is_terminal(self, s: GoState) -> bool:
        return s.passes >= 2

    def winner(self, s: GoState) -> int | None:
        if not self.is_terminal(s):
            return None
        black_area, white_area = self._area_scores(s.board)
        # Half-integer komi guarantees a non-zero margin (no draws).
        return 1 if black_area - white_area - self.komi > 0 else -1

    def encode(self, s: GoState) -> np.ndarray:
        board = np.asarray(s.board, dtype=np.float32)
        current = (board == s.player).astype(np.float32)
        opponent = (board == -s.player).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    def __str__(self, s: GoState) -> str:
        glyph = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(glyph[cell] for cell in row) for row in s.board]
        header = " ".join(str(col % 10) for col in range(self.size))
        to_move = "X" if s.player == 1 else "O"
        return "\n".join([*rows, header, f"to move: {to_move}"])

    # --- internals -------------------------------------------------------

    def _neighbors(self, point: int) -> list[int]:
        row, col = divmod(point, self.size)
        result = []
        if row > 0:
            result.append(point - self.size)
        if row < self.size - 1:
            result.append(point + self.size)
        if col > 0:
            result.append(point - 1)
        if col < self.size - 1:
            result.append(point + 1)
        return result

    def _group(self, board: list[list[int]], point: int) -> tuple[set[int], set[int]]:
        """Return the connected same-color group at `point` and its liberties."""

        color = board[point // self.size][point % self.size]
        group: set[int] = set()
        liberties: set[int] = set()
        stack = [point]
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            for neighbor in self._neighbors(current):
                value = board[neighbor // self.size][neighbor % self.size]
                if value == 0:
                    liberties.add(neighbor)
                elif value == color and neighbor not in group:
                    stack.append(neighbor)
        return group, liberties

    def _place(
        self, board: tuple[tuple[int, ...], ...], point: int, player: int
    ) -> tuple[list[list[int]], list[int]]:
        """Place `player`'s stone and remove captured opponent groups.

        Returns the mutable resulting board and the list of captured points.
        Does not check for suicide; the caller inspects the mover's liberties.
        """

        new_board = [list(row) for row in board]
        new_board[point // self.size][point % self.size] = player

        captured: list[int] = []
        for neighbor in self._neighbors(point):
            if new_board[neighbor // self.size][neighbor % self.size] == -player:
                group, liberties = self._group(new_board, neighbor)
                if not liberties:
                    for stone in group:
                        new_board[stone // self.size][stone % self.size] = 0
                    captured.extend(group)
        return new_board, captured

    def _is_legal_point(self, s: GoState, point: int) -> bool:
        if s.board[point // self.size][point % self.size] != 0:
            return False
        if point == s.ko_point:
            return False
        new_board, _ = self._place(s.board, point, s.player)
        _, liberties = self._group(new_board, point)
        return len(liberties) > 0

    def _area_scores(self, board: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
        """Tromp-Taylor area: stones plus empty regions bordered by one color."""

        black = sum(cell == 1 for row in board for cell in row)
        white = sum(cell == -1 for row in board for cell in row)

        visited: set[int] = set()
        for start in range(self.size * self.size):
            if board[start // self.size][start % self.size] != 0 or start in visited:
                continue
            region: set[int] = set()
            border_colors: set[int] = set()
            stack = [start]
            while stack:
                current = stack.pop()
                if current in region:
                    continue
                region.add(current)
                visited.add(current)
                for neighbor in self._neighbors(current):
                    value = board[neighbor // self.size][neighbor % self.size]
                    if value == 0:
                        stack.append(neighbor)
                    else:
                        border_colors.add(value)
            if border_colors == {1}:
                black += len(region)
            elif border_colors == {-1}:
                white += len(region)
        return black, white
