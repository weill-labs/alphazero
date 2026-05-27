"""Connect Four solved-ness certification harness.

The certifier compares a trained Connect Four agent against the exact solver
on a deterministic sample of non-terminal positions. Short opening positions
are always included in the sample; with the current bounded solver they may be
skipped, and they will start counting automatically once the solver can handle
them within the configured node budget.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from alphazero.arena import MCTSPlayer, RandomPlayer
from alphazero.c4_solver import NodeBudgetExceeded, solve
from alphazero.game import Game, State
from alphazero.games import ConnectFour
from alphazero.play import load_checkpoint

C4_BOARD_CELLS = 42
DEFAULT_SAMPLE_SIZE = 32
DEFAULT_SIMS = 200
DEFAULT_SOLVER_MAX_NODES = 250_000
DEFAULT_OPENING_DEPTH = 2
DEFAULT_RANDOM_MIN_PLIES = 18
DEFAULT_RANDOM_MAX_PLIES = 38
_CENTER_FIRST_COLS = (3, 2, 4, 1, 5, 0, 6)


class Net(Protocol):
    def predict(self, state_encoding: np.ndarray) -> tuple[np.ndarray, float]: ...


class Player(Protocol):
    def select_action(self, game: Game, state: State) -> int: ...


@dataclass(frozen=True)
class PositionCertification:
    solver_value: int
    optimal_moves: tuple[int, ...]
    agent_move: int
    agent_value: float
    agent_outcome: int
    policy_match: bool
    blunder: bool


@dataclass(frozen=True)
class CertificationReport:
    sampled_positions: int
    evaluated_positions: int
    skipped_positions: int
    policy_matches: int
    blunders: int
    policy_match_percent: float
    blunder_rate: float
    value_mae: float
    solved: bool
    records: tuple[PositionCertification, ...]

    def as_dict(self) -> dict[str, bool | float | int]:
        return {
            "sampled_positions": self.sampled_positions,
            "evaluated_positions": self.evaluated_positions,
            "skipped_positions": self.skipped_positions,
            "policy_matches": self.policy_matches,
            "blunders": self.blunders,
            "policy_match_percent": self.policy_match_percent,
            "blunder_rate": self.blunder_rate,
            "value_mae": self.value_mae,
            "solved": self.solved,
        }


def certify_connect_four(
    net: Net,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sims: int = DEFAULT_SIMS,
    seed: int = 0,
    solver_max_nodes: int = DEFAULT_SOLVER_MAX_NODES,
    positions: Sequence[State] | None = None,
    player: Player | None = None,
    game: ConnectFour | None = None,
    opening_depth: int = DEFAULT_OPENING_DEPTH,
) -> CertificationReport:
    """Compare a Connect Four agent's MCTS move and value to solver labels."""

    c4 = game if game is not None else ConnectFour()
    if not isinstance(c4, ConnectFour):
        raise TypeError("certify_connect_four requires a ConnectFour game")
    _validate_positive("sample_size", sample_size)
    _validate_positive("sims", sims)
    _validate_positive("solver_max_nodes", solver_max_nodes)
    if opening_depth < 0:
        raise ValueError("opening_depth must be non-negative")

    sample = (
        list(positions)
        if positions is not None
        else sample_positions(
            c4,
            sample_size=sample_size,
            seed=seed,
            opening_depth=opening_depth,
        )
    )
    agent = (
        player
        if player is not None
        else MCTSPlayer(
            net,
            num_simulations=sims,
            temperature=0.0,
            dirichlet_eps=0.0,
            seed=seed,
        )
    )

    records: list[PositionCertification] = []
    skipped_positions = 0

    for state in sample:
        if c4.is_terminal(state) or not c4.legal_moves(state):
            skipped_positions += 1
            continue

        try:
            solver_value, optimal_moves = solve(state, max_nodes=solver_max_nodes)
        except NodeBudgetExceeded:
            skipped_positions += 1
            continue
        if not optimal_moves:
            skipped_positions += 1
            continue

        agent_move = agent.select_action(c4, state)
        if agent_move not in c4.legal_moves(state):
            raise ValueError(f"agent selected illegal action {agent_move}")

        try:
            child_value, _ = solve(
                c4.apply_move(state, agent_move),
                max_nodes=solver_max_nodes,
            )
        except NodeBudgetExceeded:
            skipped_positions += 1
            continue

        _, agent_value = net.predict(c4.encode(state))
        agent_outcome = -child_value
        policy_match = agent_move in optimal_moves
        records.append(
            PositionCertification(
                solver_value=solver_value,
                optimal_moves=tuple(optimal_moves),
                agent_move=agent_move,
                agent_value=float(agent_value),
                agent_outcome=agent_outcome,
                policy_match=policy_match,
                blunder=agent_outcome < solver_value,
            )
        )

    return _report(
        sampled_positions=len(sample),
        skipped_positions=skipped_positions,
        records=records,
    )


def sample_positions(
    game: ConnectFour,
    *,
    sample_size: int,
    seed: int,
    opening_depth: int = DEFAULT_OPENING_DEPTH,
    random_min_plies: int = DEFAULT_RANDOM_MIN_PLIES,
    random_max_plies: int = DEFAULT_RANDOM_MAX_PLIES,
) -> list[State]:
    """Return a deterministic mix of random self-play and short openings."""

    if not isinstance(game, ConnectFour):
        raise TypeError("sample_positions requires ConnectFour")
    _validate_positive("sample_size", sample_size)
    if opening_depth < 0:
        raise ValueError("opening_depth must be non-negative")
    if random_min_plies < 0:
        raise ValueError("random_min_plies must be non-negative")
    if random_max_plies < random_min_plies:
        raise ValueError("random_max_plies must be >= random_min_plies")

    opening_quota = min(sample_size // 4, sample_size)
    random_quota = sample_size - opening_quota
    rng = np.random.default_rng(seed)
    seen: set[State] = set()
    sample: list[State] = []

    sample.extend(
        _random_self_play_positions(
            game,
            count=random_quota,
            seed=seed,
            rng=rng,
            random_min_plies=random_min_plies,
            random_max_plies=random_max_plies,
            seen=seen,
        )
    )

    for state in _opening_positions(game, max_depth=opening_depth):
        if len(sample) >= sample_size:
            break
        if state not in seen:
            seen.add(state)
            sample.append(state)

    if len(sample) < sample_size:
        sample.extend(
            _random_self_play_positions(
                game,
                count=sample_size - len(sample),
                seed=seed + 1,
                rng=rng,
                random_min_plies=0,
                random_max_plies=C4_BOARD_CELLS - 1,
                seen=seen,
            )
        )

    return sample[:sample_size]


def _report(
    *,
    sampled_positions: int,
    skipped_positions: int,
    records: Sequence[PositionCertification],
) -> CertificationReport:
    evaluated_positions = len(records)
    policy_matches = sum(1 for record in records if record.policy_match)
    blunders = sum(1 for record in records if record.blunder)
    value_errors = [
        abs(float(record.agent_value) - float(record.solver_value))
        for record in records
    ]
    value_mae = float(np.mean(value_errors)) if value_errors else 0.0
    policy_match_percent = (
        100.0 * policy_matches / evaluated_positions if evaluated_positions else 0.0
    )
    blunder_rate = blunders / evaluated_positions if evaluated_positions else 0.0
    return CertificationReport(
        sampled_positions=sampled_positions,
        evaluated_positions=evaluated_positions,
        skipped_positions=skipped_positions,
        policy_matches=policy_matches,
        blunders=blunders,
        policy_match_percent=policy_match_percent,
        blunder_rate=blunder_rate,
        value_mae=value_mae,
        solved=evaluated_positions > 0 and blunder_rate == 0.0,
        records=tuple(records),
    )


def _random_self_play_positions(
    game: ConnectFour,
    *,
    count: int,
    seed: int,
    rng: np.random.Generator,
    random_min_plies: int,
    random_max_plies: int,
    seen: set[State],
) -> list[State]:
    if count <= 0:
        return []

    positions: list[State] = []
    player = RandomPlayer(seed=seed)
    max_attempts = max(count * 100, 100)

    for _ in range(max_attempts):
        state = game.initial_state()
        target_plies = int(rng.integers(random_min_plies, random_max_plies + 1))
        for _ in range(target_plies):
            if game.is_terminal(state):
                break
            state = game.apply_move(state, player.select_action(game, state))

        if game.is_terminal(state) or not game.legal_moves(state) or state in seen:
            continue

        seen.add(state)
        positions.append(state)
        if len(positions) == count:
            break

    return positions


def _opening_positions(game: ConnectFour, *, max_depth: int) -> list[State]:
    positions = [game.initial_state()]
    frontier = [game.initial_state()]
    for _ in range(max_depth):
        next_frontier: list[State] = []
        for state in frontier:
            for action in _ordered_legal_moves(game, state):
                child = game.apply_move(state, action)
                if not game.is_terminal(child):
                    positions.append(child)
                    next_frontier.append(child)
        frontier = next_frontier
    return positions


def _ordered_legal_moves(game: ConnectFour, state: State) -> list[int]:
    legal = set(game.legal_moves(state))
    return [action for action in _CENTER_FIRST_COLS if action in legal]


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Certify a Connect Four checkpoint against exact solver labels."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-max-nodes", type=int, default=DEFAULT_SOLVER_MAX_NODES
    )
    parser.add_argument("--opening-depth", type=int, default=DEFAULT_OPENING_DEPTH)
    args = parser.parse_args(argv)

    game = ConnectFour()
    net = load_checkpoint(args.checkpoint, game)
    report = certify_connect_four(
        net,
        sample_size=args.sample_size,
        sims=args.sims,
        seed=args.seed,
        solver_max_nodes=args.solver_max_nodes,
        game=game,
        opening_depth=args.opening_depth,
    )
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
