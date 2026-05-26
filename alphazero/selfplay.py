"""Self-play data generation for the AlphaZero training pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Protocol

import numpy as np

from alphazero.game import Game
from alphazero.mcts import MCTS, TimingHook


class _Net(Protocol):
    """Subset of the network API used by MCTS during self-play."""

    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]: ...


TemperatureSchedule = Callable[[int], float] | Sequence[float] | float
SelfPlayExample = tuple[np.ndarray, np.ndarray, int]


def play_game(
    net: _Net,
    game: Game,
    mcts_cfg: Mapping[str, object] | None,
    temperature_schedule: TemperatureSchedule,
    *,
    timing_hook: TimingHook | None = None,
) -> list[SelfPlayExample]:
    """Play one self-play game and return ``(encoded_state, pi, z)`` examples."""

    mcts_kwargs = dict(mcts_cfg or {})
    if timing_hook is not None:
        mcts_kwargs["timing_hook"] = timing_hook
    mcts = MCTS(net, game, **mcts_kwargs)
    state = game.initial_state()
    pending: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_index = 0

    while not game.is_terminal(state):
        mover = game.current_player(state)
        encoded_state = game.encode(state).copy()
        pi = mcts.run(state, add_noise=True)
        pi = _normalize_for_legal_moves(pi, game.legal_moves(state), game.action_size)
        pending.append((encoded_state, pi.copy(), mover))

        temperature = _temperature_at(temperature_schedule, move_index)
        action = mcts.select_action(pi, temperature=temperature)
        state = game.apply_move(state, action)
        move_index += 1

    outcome = game.winner(state)
    if outcome is None:
        raise RuntimeError("self-play ended without a terminal winner/draw value")

    examples: list[SelfPlayExample] = []
    for encoded_state, pi, mover in pending:
        if outcome == 0:
            z = 0
        elif outcome == mover:
            z = 1
        else:
            z = -1
        examples.append((encoded_state, pi, z))
    return examples


def _temperature_at(schedule: TemperatureSchedule, move_index: int) -> float:
    if callable(schedule):
        return float(schedule(move_index))
    if isinstance(schedule, Real):
        return float(schedule)
    if not schedule:
        raise ValueError("temperature_schedule must not be empty")
    if move_index < len(schedule):
        return float(schedule[move_index])
    return float(schedule[-1])


def _normalize_for_legal_moves(
    pi: np.ndarray, legal_moves: list[int], action_size: int
) -> np.ndarray:
    policy = np.asarray(pi, dtype=np.float64).copy()
    if policy.shape != (action_size,):
        raise ValueError(
            f"expected MCTS policy shape ({action_size},), got {policy.shape}"
        )

    legal_mask = np.zeros(action_size, dtype=bool)
    legal_mask[legal_moves] = True
    policy[~legal_mask] = 0.0
    total = policy.sum()
    if total > 0:
        policy /= total
        return policy

    if not legal_moves:
        return policy
    policy[legal_moves] = 1.0 / len(legal_moves)
    return policy
