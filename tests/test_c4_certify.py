from __future__ import annotations

import subprocess
import sys

import jax
import numpy as np

from alphazero.c4_certify import (
    JaxMCTSAgent,
    certify_connect_four,
    pgx_state_to_solver_state,
    sample_positions,
    solver_state_to_pgx_state,
)
from alphazero.c4_solver import solve
from alphazero.games.connectfour import ConnectFour, ConnectFourState
from jaxzero.net import AlphaZeroNetConfig, create_model
from jaxzero.train import save_checkpoint


_FORCED_BLOCK_DRAW = [
    0,
    4,
    3,
    2,
    6,
    1,
    4,
    1,
    0,
    5,
    5,
    1,
    4,
    0,
    4,
    5,
    6,
    4,
    4,
    3,
    3,
    2,
    2,
    2,
    0,
    3,
]


class SolverOracleAgent:
    def move(self, state: ConnectFourState) -> int:
        _, optimal_moves = solve(state)
        return optimal_moves[0]

    def value(self, state: ConnectFourState) -> float:
        solver_value, _ = solve(state)
        return float(solver_value)


class FixedActionAgent:
    def __init__(self, action: int) -> None:
        self.action = action

    def move(self, state: ConnectFourState) -> int:
        game = ConnectFour()
        if self.action not in game.legal_moves(state):
            raise ValueError(f"fixed action {self.action} is illegal")
        return self.action

    def value(self, state: ConnectFourState) -> float:
        del state
        return 0.0


def _play(moves: list[int]):
    game = ConnectFour()
    state = game.initial_state()
    for move in moves:
        state = game.apply_move(state, move)
    return game, state


def test_perfect_vs_solver_toy_case_scores_zero_blunders() -> None:
    _, state = _play(_FORCED_BLOCK_DRAW)

    report = certify_connect_four(
        SolverOracleAgent(),
        positions=[state],
    )

    assert report.evaluated_positions == 1
    assert np.isclose(report.policy_match_percent, 100.0)
    assert np.isclose(report.blunder_rate, 0.0)
    assert np.isclose(report.value_mae, 0.0)
    assert report.solved


def test_known_bad_move_is_flagged_as_blunder() -> None:
    _, state = _play(_FORCED_BLOCK_DRAW)

    report = certify_connect_four(
        FixedActionAgent(0),
        positions=[state],
    )

    assert report.evaluated_positions == 1
    assert np.isclose(report.policy_match_percent, 0.0)
    assert report.blunders == 1
    assert np.isclose(report.blunder_rate, 1.0)
    assert not report.solved


def test_position_sampling_is_deterministic() -> None:
    first = sample_positions(sample_size=8, seed=123)
    second = sample_positions(sample_size=8, seed=123)

    assert first == second
    assert len(first) == 8


def test_pgx_adapter_round_trips_solver_state() -> None:
    _, state = _play([3, 2, 3, 2, 4])

    pgx_state = solver_state_to_pgx_state(state)
    restored = pgx_state_to_solver_state(pgx_state)

    assert restored == state
    assert int(jax.device_get(pgx_state._x.color)) == 1
    assert pgx_state.observation.shape == (6, 7, 2)
    assert pgx_state.legal_action_mask.tolist() == [True] * 7


def test_jax_mcts_agent_loads_checkpoint_and_selects_legal_move(tmp_path) -> None:
    game = ConnectFour()
    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2),
        action_size=game.action_size,
        channels=4,
        num_res_blocks=0,
    )
    checkpoint = tmp_path / "jaxzero.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    agent = JaxMCTSAgent.from_checkpoint(checkpoint, sims=1, seed=0)
    state = game.initial_state()

    move = agent.move(state)
    value = agent.value(state)

    assert move in game.legal_moves(state)
    assert np.isfinite(value)


def test_c4_certify_imports_do_not_load_torch() -> None:
    code = "import sys, alphazero.c4_certify; raise SystemExit('torch' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
