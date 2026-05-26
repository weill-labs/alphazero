"""Network-guided Monte Carlo Tree Search (PUCT), the AlphaZero planner.

MCTS turns the network's raw policy/value estimates into a stronger move
distribution by simulating many lookahead paths. Each edge accumulates the
statistics ``N`` (visit count), ``W`` (total value), ``Q`` (mean value = W/N),
and ``P`` (network prior). Selection uses the PUCT rule; leaves are collected
into leaf-parallel batches for network evaluation; values are backed up with a
per-ply sign flip because the game is zero-sum. See docs/ARCHITECTURE.md for
the contract.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from alphazero.game import Game, State


class _Net(Protocol):
    """Subset of the network API that MCTS depends on (see AlphaZeroNet)."""

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]: ...


TimingHook = Callable[[str, float], None]
_SearchPath = list[tuple["_Node", int]]


class _Node:
    """A searched game state. Edge stats are stored per legal action."""

    __slots__ = (
        "state",
        "player",
        "is_terminal",
        "expanded",
        "P",
        "N",
        "W",
        "children",
    )

    def __init__(self, state: State, player: int, is_terminal: bool) -> None:
        self.state = state
        self.player = player
        self.is_terminal = is_terminal
        self.expanded = False
        self.P: dict[int, float] = {}  # action -> prior
        self.N: dict[int, int] = {}  # action -> visit count
        self.W: dict[int, float] = {}  # action -> total value (parent-perspective)
        self.children: dict[int, _Node] = {}


class MCTS:
    def __init__(
        self,
        net: _Net,
        game: Game,
        c_puct: float = 1.5,
        num_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        seed: int | None = None,
        timing_hook: TimingHook | None = None,
        batch_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.net = net
        self.game = game
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)
        self.timing_hook = timing_hook
        self.batch_size = batch_size

    # -- public API ----------------------------------------------------------

    def run(self, state: State, add_noise: bool = False) -> np.ndarray:
        """Search from `state` and return a visit-count policy.

        The returned array has length ``game.action_size``, sums to 1 over
        legal moves, and stays zero on illegal moves. Set ``add_noise=True``
        during self-play to mix Dirichlet exploration noise into root priors.
        """
        root = self._make_node(state)
        pi = np.zeros(self.game.action_size, dtype=np.float64)
        if root.is_terminal:
            return pi  # no moves to search from a finished game

        self._expand(root)
        if add_noise:
            self._add_dirichlet_noise(root)

        completed = 0
        while completed < self.num_simulations:
            completed += self._simulate_batch(root, self.num_simulations - completed)

        total = sum(root.N.values())
        if total > 0:
            for a, n in root.N.items():
                pi[a] = n / total
        return pi

    def select_action(
        self,
        pi: np.ndarray,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Pick an action from a visit-count policy `pi`.

        ``temperature`` reshapes the distribution: ``tau -> 0`` is greedy
        (argmax of visits, ties broken randomly); ``tau = 1`` samples
        proportionally to visits; larger values flatten toward uniform.
        """
        rng = rng if rng is not None else self.rng
        weights = np.asarray(pi, dtype=np.float64)
        if temperature <= 1e-8:
            best = weights.max()
            winners = np.flatnonzero(weights >= best - 1e-12)
            return int(rng.choice(winners))
        scaled = np.zeros_like(weights)
        positive = weights > 0
        scaled[positive] = weights[positive] ** (1.0 / temperature)
        total = scaled.sum()
        if total <= 0:
            # Degenerate input (all-zero pi): fall back to uniform.
            scaled = np.ones_like(weights) / weights.size
        else:
            scaled /= total
        return int(rng.choice(weights.size, p=scaled))

    # -- internals -----------------------------------------------------------

    def _make_node(self, state: State) -> _Node:
        return _Node(
            state=state,
            player=self.game.current_player(state),
            is_terminal=self.game.is_terminal(state),
        )

    def _expand(self, node: _Node) -> float:
        """Evaluate one node, set priors, and return current-player value."""
        probs_batch, values = self._predict_batch([self.game.encode(node.state)])
        self._expand_with_prediction(node, probs_batch[0])
        return float(values[0])

    def _expand_with_prediction(self, node: _Node, probs: np.ndarray) -> None:
        """Set masked/renormalized priors from an already computed policy."""
        policy = np.asarray(probs, dtype=np.float64)
        if policy.shape != (self.game.action_size,):
            raise ValueError(
                f"expected policy shape ({self.game.action_size},), got {policy.shape}"
            )

        legal = self.game.legal_moves(node.state)
        masked = {a: max(float(policy[a]), 0.0) for a in legal}
        total = sum(masked.values())
        if total > 0:
            node.P = {a: p / total for a, p in masked.items()}
        else:
            # Network assigned ~0 mass to every legal move: use a uniform prior.
            node.P = {a: 1.0 / len(legal) for a in legal}
        node.N = {a: 0 for a in legal}
        node.W = {a: 0.0 for a in legal}
        node.expanded = True

    def _add_dirichlet_noise(self, node: _Node) -> None:
        if self.dirichlet_eps <= 0:
            return
        actions = list(node.P.keys())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_eps
        for action, n in zip(actions, noise):
            node.P[action] = (1.0 - eps) * node.P[action] + eps * float(n)

    def _puct_select(self, node: _Node) -> int:
        """Return the action maximizing Q + c_puct * P * sqrt(sum N) / (1 + N)."""
        sqrt_total = math.sqrt(sum(node.N.values()))
        best_action = -1
        best_score = -math.inf
        for action, prior in node.P.items():
            n = node.N[action]
            q = node.W[action] / n if n > 0 else 0.0
            u = self.c_puct * prior * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(self, root: _Node) -> None:
        self._simulate_batch(root, 1)

    def _simulate_batch(self, root: _Node, max_simulations: int) -> int:
        """Run up to one leaf-parallel batch and return completed simulations."""
        pending: list[tuple[_Node, _SearchPath]] = []
        pending_node_ids: set[int] = set()
        completed = 0

        while completed < max_simulations and len(pending) < self.batch_size:
            node, path = self._select_leaf(root)

            if node.is_terminal:
                value = self.game.winner(node.state) * node.player
                self._backup(path, value)
                completed += 1
                continue

            node_id = id(node)
            if node_id in pending_node_ids:
                if pending:
                    break
                # Defensive fallback for unusual game/tree shapes.
                self._reserve_path(path)
                pending.append((node, path))
                completed += 1
                continue

            self._reserve_path(path)
            pending.append((node, path))
            pending_node_ids.add(node_id)
            completed += 1

        if pending:
            encodings = [self.game.encode(node.state) for node, _ in pending]
            probs_batch, values = self._predict_batch(encodings)
            for (node, path), probs, value in zip(
                pending, probs_batch, values, strict=True
            ):
                self._unreserve_path(path)
                self._expand_with_prediction(node, probs)
                self._backup(path, float(value))

        return completed

    def _select_leaf(self, root: _Node) -> tuple[_Node, _SearchPath]:
        node = root
        path: _SearchPath = []
        # Selection: descend via PUCT until we reach a leaf (unexpanded or terminal).
        while node.expanded and not node.is_terminal:
            action = self._puct_select(node)
            if action not in node.children:
                child_state = self.game.apply_move(node.state, action)
                node.children[action] = self._make_node(child_state)
            path.append((node, action))
            node = node.children[action]
        return node, path

    def _predict_batch(
        self, encodings: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not encodings:
            raise ValueError("encodings must not be empty")

        batch = np.stack(encodings, axis=0)
        started = time.perf_counter() if self.timing_hook is not None else 0.0
        predictor = getattr(self.net, "predict_batch", None)
        if callable(predictor):
            probs, values = predictor(batch)
        else:
            predictions = [self.net.predict(encoding) for encoding in encodings]
            probs = np.stack([policy for policy, _ in predictions], axis=0)
            values = np.asarray([value for _, value in predictions], dtype=np.float64)

        if self.timing_hook is not None:
            self._record_network_inference(
                time.perf_counter() - started, len(encodings)
            )

        probs_array = np.asarray(probs, dtype=np.float64)
        values_array = np.asarray(values, dtype=np.float64)
        expected_policy_shape = (len(encodings), self.game.action_size)
        if probs_array.shape != expected_policy_shape:
            raise ValueError(
                f"expected policy batch shape {expected_policy_shape}, "
                f"got {probs_array.shape}"
            )
        if values_array.shape != (len(encodings),):
            raise ValueError(
                f"expected value batch shape ({len(encodings)},), "
                f"got {values_array.shape}"
            )
        return probs_array, values_array

    def _record_network_inference(self, seconds: float, positions: int) -> None:
        if self.timing_hook is None:
            return
        # Benchmark counts are position-evals; seconds are charged once per batch.
        self.timing_hook("network_inference", seconds)
        for _ in range(positions - 1):
            self.timing_hook("network_inference", 0.0)

    def _reserve_path(self, path: _SearchPath) -> None:
        for parent, action in path:
            parent.N[action] += 1
            parent.W[action] -= 1.0

    def _unreserve_path(self, path: _SearchPath) -> None:
        for parent, action in path:
            parent.N[action] -= 1
            parent.W[action] += 1.0

    def _backup(self, path: _SearchPath, value: float) -> None:
        """Propagate `value` up the path, negating once per ply (zero-sum)."""
        for parent, action in reversed(path):
            value = -value  # flip to the perspective of `parent`'s player
            parent.N[action] += 1
            parent.W[action] += value
