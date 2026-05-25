"""Abstract, game-agnostic interface for the AlphaZero pipeline.

MCTS, self-play, and training interact with a game ONLY through this interface,
so supporting a new game means writing a new `Game` subclass and nothing else.
See docs/ARCHITECTURE.md for the binding integration contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# A State is an opaque, immutable, hashable value owned by each Game
# implementation. The pipeline never inspects its internals; it only round-trips
# states through the methods below. Hashability lets MCTS key nodes by state.
State = Any


class Game(ABC):
    """Rules of a two-player, zero-sum, perfect-information game.

    Subclasses set the three class attributes (so the network can size its
    input/output without knowing the game) and implement the methods.
    """

    action_size: int  # number of distinct actions (policy head width)
    board_shape: tuple[int, int]  # (H, W) of the encoded board
    num_planes: int  # feature planes produced by encode()

    @abstractmethod
    def initial_state(self) -> State:
        """Return the starting state, with player +1 to move."""

    @abstractmethod
    def current_player(self, s: State) -> int:
        """Return +1 or -1: the player to move in `s`."""

    @abstractmethod
    def legal_moves(self, s: State) -> list[int]:
        """Return legal action indices in `s` (empty list if terminal)."""

    @abstractmethod
    def apply_move(self, s: State, a: int) -> State:
        """Return the NEW state reached by playing action `a`.

        Must not mutate `s` (states are immutable so MCTS can reuse them).
        """

    @abstractmethod
    def is_terminal(self, s: State) -> bool:
        """Return True if `s` is an end state (win, loss, or draw)."""

    @abstractmethod
    def winner(self, s: State) -> int | None:
        """Return +1/-1 for the winner, 0 for a draw, None if not over."""

    @abstractmethod
    def encode(self, s: State) -> np.ndarray:
        """Return a canonical ``(num_planes, H, W)`` float32 encoding of `s`.

        Canonical = from the perspective of the player to move: that player's
        pieces on plane 0, the opponent's on plane 1. This perspective
        normalization is what lets a single network evaluate both players.
        """

    @abstractmethod
    def __str__(self, s: State) -> str:
        """Return a human-readable rendering of `s` for debugging."""
