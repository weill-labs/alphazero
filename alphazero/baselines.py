"""Game-agnostic baseline players."""

from __future__ import annotations

import math

import numpy as np

from alphazero.game import Game, State

_WIN_SCORE = 1_000_000.0


class NegamaxPlayer:
    """Depth-limited negamax player with alpha-beta pruning."""

    def __init__(self, depth: int = 4) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth

    def select_action(self, game: Game, state: State) -> int:
        legal = sorted(game.legal_moves(state))
        if not legal:
            raise ValueError("cannot select an action for a terminal state")

        best_action = legal[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in legal:
            score = -self._negamax(
                game,
                game.apply_move(state, action),
                self.depth - 1,
                -beta,
                -alpha,
            )
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action

    def _negamax(
        self,
        game: Game,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        terminal_value = self._terminal_value(game, state, depth)
        if terminal_value is not None:
            return terminal_value
        if depth == 0:
            return self._heuristic(game, state)

        value = -math.inf
        for action in sorted(game.legal_moves(state)):
            score = -self._negamax(
                game,
                game.apply_move(state, action),
                depth - 1,
                -beta,
                -alpha,
            )
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value

    def _terminal_value(self, game: Game, state: State, depth: int) -> float | None:
        if not game.is_terminal(state):
            return None

        winner = game.winner(state)
        if winner is None or winner == 0:
            return 0.0

        current_player = game.current_player(state)
        if winner == current_player:
            return _WIN_SCORE + depth
        return -_WIN_SCORE - depth

    def _heuristic(self, game: Game, state: State) -> float:
        encoded = np.asarray(game.encode(state), dtype=np.float32)
        if encoded.ndim < 3 or encoded.shape[0] < 2:
            return 0.0

        own_count = float(encoded[0].sum())
        opponent_count = float(encoded[1].sum())
        board_cells = max(int(np.prod(encoded.shape[1:])), 1)
        material = (own_count - opponent_count) / board_cells
        mobility = len(game.legal_moves(state)) / max(game.action_size, 1)
        return material + 0.01 * mobility
