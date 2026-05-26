"""Exact Connect Four solver for evaluation-only oracle checks.

The internal representation follows the compact Pascal Pons bitboard layout:
each column uses seven bits, with six playable cells plus one sentinel bit.
Bit index ``col * 7 + row`` maps to a board cell where ``row=0`` is the
bottom playable row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alphazero.games.connectfour import ConnectFour, ConnectFourState

_HEIGHT = 6
_WIDTH = 7
_STRIDE = _HEIGHT + 1
_BOARD_CELLS = _HEIGHT * _WIDTH
_CENTER_FIRST_COLS = (3, 2, 4, 1, 5, 0, 6)
_DEFAULT_MAX_NODES = 5_000_000
_MAX_SCORE = (_BOARD_CELLS + 1) // 2

_BOTTOM_MASKS = tuple(1 << (col * _STRIDE) for col in range(_WIDTH))
_TOP_MASKS = tuple(1 << ((_HEIGHT - 1) + col * _STRIDE) for col in range(_WIDTH))
_COLUMN_MASKS = tuple(((1 << _HEIGHT) - 1) << (col * _STRIDE) for col in range(_WIDTH))
_BOTTOM_MASK = sum(_BOTTOM_MASKS)
_BOARD_MASK = sum(_COLUMN_MASKS)


class NodeBudgetExceeded(RuntimeError):
    """Raised when exact search exceeds the configured node budget."""


@dataclass
class _Position:
    current: int
    opponent: int
    mask: int
    moves_played: int


@dataclass
class _Solver:
    max_nodes: int
    table: dict[tuple[int, int], int] = field(default_factory=dict)
    nodes: int = 0

    def score(
        self,
        current: int,
        opponent: int,
        mask: int,
        moves_played: int,
        alpha: int,
        beta: int,
    ) -> int:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise NodeBudgetExceeded(
                f"Connect Four solve exceeded node budget of {self.max_nodes}"
            )

        if _has_won(opponent):
            return -((_BOARD_CELLS - moves_played + 2) // 2)
        if _can_win_next(current, mask):
            return (_BOARD_CELLS + 1 - moves_played) // 2

        possible_moves = _possible_non_losing_moves(current, opponent, mask)
        if possible_moves == 0:
            return -((_BOARD_CELLS - moves_played) // 2)
        if moves_played >= _BOARD_CELLS - 2:
            return 0

        min_score = -((_BOARD_CELLS - 2 - moves_played) // 2)
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha

        max_score = (_BOARD_CELLS - 1 - moves_played) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta

        key = (current, mask)
        cached_upper_bound = self.table.get(key)
        if cached_upper_bound is not None and beta > cached_upper_bound:
            beta = cached_upper_bound
            if alpha >= beta:
                return beta

        for move in _ordered_move_bits(possible_moves, current, mask):
            value = -self.score(
                opponent,
                current | move,
                mask | move,
                moves_played + 1,
                -beta,
                -alpha,
            )
            if value >= beta:
                return value
            alpha = max(alpha, value)

        self.table[key] = alpha
        return alpha


def solve(
    state: ConnectFourState,
    *,
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> tuple[int, list[int]]:
    """Return the perfect-play value and optimal columns for ``state``.

    ``value`` is from the player-to-move perspective: ``+1`` means a forced
    win, ``0`` means a draw, and ``-1`` means a forced loss.
    """

    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")

    game = ConnectFour()
    winner = game.winner(state)
    if winner is not None:
        if winner == 0:
            return 0, []
        return (1 if winner == state.player else -1), []

    position = _from_state(state)
    legal = [col for col in _CENTER_FIRST_COLS if position.mask & _TOP_MASKS[col] == 0]
    if not legal:
        return 0, []

    solver = _Solver(max_nodes=max_nodes)
    best_value = -2
    optimal_moves: list[int] = []
    for col in legal:
        move = _move_bit(position.mask, col)
        solver.table.clear()
        child_score = solver.score(
            position.opponent,
            position.current | move,
            position.mask | move,
            position.moves_played + 1,
            -_MAX_SCORE,
            _MAX_SCORE,
        )
        value = -_score_to_value(child_score)
        if value > best_value:
            best_value = value
            optimal_moves = [col]
        elif value == best_value:
            optimal_moves.append(col)

    return best_value, sorted(optimal_moves)


def _from_state(state: ConnectFourState) -> _Position:
    current = 0
    opponent = 0
    mask = 0
    moves_played = 0

    for row_index, row in enumerate(state.board):
        bit_row = _HEIGHT - 1 - row_index
        for col, cell in enumerate(row):
            if cell == 0:
                continue
            bit = 1 << (col * _STRIDE + bit_row)
            mask |= bit
            moves_played += 1
            if cell == state.player:
                current |= bit
            else:
                opponent |= bit

    return _Position(
        current=current,
        opponent=opponent,
        mask=mask,
        moves_played=moves_played,
    )


def _score_to_value(score: int) -> int:
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def _has_won(position: int) -> bool:
    vertical = position & (position >> 1)
    if vertical & (vertical >> 2):
        return True

    horizontal = position & (position >> _STRIDE)
    if horizontal & (horizontal >> (2 * _STRIDE)):
        return True

    diagonal_up = position & (position >> _HEIGHT)
    if diagonal_up & (diagonal_up >> (2 * _HEIGHT)):
        return True

    diagonal_down = position & (position >> (_HEIGHT + 2))
    return bool(diagonal_down & (diagonal_down >> (2 * (_HEIGHT + 2))))


def _winning_positions(position: int, mask: int) -> int:
    winning = (position << 1) & (position << 2) & (position << 3)

    pair = (position << _STRIDE) & (position << (2 * _STRIDE))
    winning |= pair & (position << (3 * _STRIDE))
    winning |= pair & (position >> _STRIDE)
    pair = (position >> _STRIDE) & (position >> (2 * _STRIDE))
    winning |= pair & (position << _STRIDE)
    winning |= pair & (position >> (3 * _STRIDE))

    pair = (position << _HEIGHT) & (position << (2 * _HEIGHT))
    winning |= pair & (position << (3 * _HEIGHT))
    winning |= pair & (position >> _HEIGHT)
    pair = (position >> _HEIGHT) & (position >> (2 * _HEIGHT))
    winning |= pair & (position << _HEIGHT)
    winning |= pair & (position >> (3 * _HEIGHT))

    diagonal_stride = _HEIGHT + 2
    pair = (position << diagonal_stride) & (position << (2 * diagonal_stride))
    winning |= pair & (position << (3 * diagonal_stride))
    winning |= pair & (position >> diagonal_stride)
    pair = (position >> diagonal_stride) & (position >> (2 * diagonal_stride))
    winning |= pair & (position << diagonal_stride)
    winning |= pair & (position >> (3 * diagonal_stride))

    return winning & _BOARD_MASK & ~mask


def _possible(mask: int) -> int:
    return (mask + _BOTTOM_MASK) & _BOARD_MASK


def _can_win_next(position: int, mask: int) -> bool:
    return bool(_winning_positions(position, mask) & _possible(mask))


def _possible_non_losing_moves(current: int, opponent: int, mask: int) -> int:
    possible = _possible(mask)
    opponent_wins = _winning_positions(opponent, mask)
    forced = possible & opponent_wins

    if forced:
        if forced & (forced - 1):
            return 0
        possible = forced

    return possible & ~(opponent_wins >> 1)


def _move_bit(mask: int, col: int) -> int:
    return (mask + _BOTTOM_MASKS[col]) & _COLUMN_MASKS[col]


def _ordered_move_bits(possible_moves: int, current: int, mask: int) -> list[int]:
    moves = [
        possible_moves & _COLUMN_MASKS[col]
        for col in _CENTER_FIRST_COLS
        if possible_moves & _COLUMN_MASKS[col]
    ]
    return sorted(
        moves,
        key=lambda move: _winning_positions(current | move, mask | move).bit_count(),
        reverse=True,
    )
