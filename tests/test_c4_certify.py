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


def test_make_solver_evaluator_returns_blunder_policy_value_and_regret_keys(
    tmp_path,
) -> None:
    """The inline solver-anchored evaluator must surface the headline blunder
    rate, the value-MAE bottleneck, policy match, AND the regret signals
    (WDL + Pons-score) used for low-variance run-vs-run comparison."""
    from alphazero.c4_certify import make_solver_evaluator
    from jaxzero.train import load_checkpoint, save_checkpoint

    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2), action_size=7, channels=4, num_res_blocks=0
    )
    checkpoint = tmp_path / "jaxzero.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    model = load_checkpoint(checkpoint)

    evaluator = make_solver_evaluator(sample_size=4, sims=1, seed=0)
    metrics = evaluator(model)

    assert set(metrics) == {
        "eval/c4_blunder_rate",
        "eval/c4_policy_match",
        "eval/c4_value_mae",
        "eval/c4_wdl_regret",
        "eval/c4_score_regret",
        "eval/c4_score_blunder_rate",
    }
    assert 0.0 <= metrics["eval/c4_blunder_rate"] <= 1.0
    assert 0.0 <= metrics["eval/c4_policy_match"] <= 1.0
    assert metrics["eval/c4_value_mae"] >= 0.0  # MAE is non-negative
    # WDL regret is a mean over {0,1,2}; score regret and strong rate >= 0.
    assert 0.0 <= metrics["eval/c4_wdl_regret"] <= 2.0
    assert metrics["eval/c4_score_regret"] >= 0.0
    assert 0.0 <= metrics["eval/c4_score_blunder_rate"] <= 1.0
    # Strong mode (any non-optimal-score move) is at least as strict as weak.
    assert metrics["eval/c4_score_blunder_rate"] >= metrics["eval/c4_blunder_rate"]


def test_perfect_agent_has_zero_regret() -> None:
    """A solver-oracle agent (picks the first WDL-optimal move) never changes
    the game outcome, so weak blunders and WDL regret are zero."""
    _, state = _play(_FORCED_BLOCK_DRAW)
    report = certify_connect_four(SolverOracleAgent(), positions=[state])

    assert report.mean_wdl_regret == 0.0
    assert report.blunders == 0
    assert np.isclose(report.blunder_rate, 0.0)


def test_regret_is_nonnegative_and_weak_blunders_subset_of_strong() -> None:
    """Weak (outcome) blunders are a subset of strong (score) blunders: changing
    the W/D/L result always trips the score blunder too, but not vice versa (a
    slower-than-fastest win is a strong blunder with wdl_regret == 0)."""
    _, state = _play(_FORCED_BLOCK_DRAW)
    report = certify_connect_four(FixedActionAgent(0), positions=[state])

    for record in report.records:
        assert record.score_regret >= 0
        assert 0 <= record.wdl_regret <= 2
        # A weak blunder implies a strong blunder.
        if record.blunder:
            assert record.score_blunder
    assert report.blunders <= report.score_blunders


def test_score_regret_distinguishes_slower_wins_from_optimal() -> None:
    """With distance-aware solver scores, score-regret must be able to exceed
    WDL-regret: a move that still wins but slower than the fastest win is a
    strong blunder (score_regret > 0) with no WDL change (wdl_regret == 0).

    Build a position by scanning legal moves for one where the agent's move
    keeps the win but is not score-optimal; assert the metric separates them.
    """
    from alphazero.c4_solver import solve_with_score

    game = ConnectFour()
    # Find any winning position with >1 legal move and a non-fastest winning
    # move, by random search seeded deterministically.
    rng = np.random.default_rng(0)
    found = False
    for _ in range(400):
        state = game.initial_state()
        for _ in range(int(rng.integers(4, 20))):
            if game.is_terminal(state):
                break
            legal = game.legal_moves(state)
            if not legal:
                break
            state = game.apply_move(state, int(rng.choice(legal)))
        if game.is_terminal(state) or not game.legal_moves(state):
            continue
        value, _, score = solve_with_score(state)
        if value <= 0:
            continue  # only winning positions have faster/slower win choices
        # Look for a still-winning move whose score is below the optimum.
        for move in game.legal_moves(state):
            child_v, _, child_s = solve_with_score(game.apply_move(state, move))
            if -child_v == 1 and -child_s < score:  # still a win, but slower
                report = certify_connect_four(FixedActionAgent(move), positions=[state])
                rec = report.records[0]
                assert rec.wdl_regret == 0  # outcome preserved (still a win)
                assert rec.score_regret > 0  # but slower than optimal
                assert rec.score_blunder and not rec.blunder
                found = True
                break
        if found:
            break

    assert found, "no slower-win position found in search budget"


def test_eval_set_roundtrip_and_paired_certification(tmp_path) -> None:
    """A saved eval set reloads to identical positions and yields an identical
    report to certifying on the in-memory positions (paired comparison basis)."""
    from alphazero.c4_certify import load_eval_set, save_eval_set

    positions = sample_positions(sample_size=6, seed=7)
    path = tmp_path / "eval_set.json"
    save_eval_set(positions, path)
    reloaded = load_eval_set(path)

    assert reloaded == positions

    agent = SolverOracleAgent()
    from_disk = certify_connect_four(agent, positions=reloaded)
    in_memory = certify_connect_four(agent, positions=positions)
    assert from_disk.as_dict() == in_memory.as_dict()


def test_build_eval_set_cli_creates_loadable_file(tmp_path) -> None:
    """The --build-eval-set CLI path writes a file that load_eval_set accepts."""
    from alphazero.c4_certify import load_eval_set, main

    path = tmp_path / "set.json"
    rc = main(["--build-eval-set", str(path), "--sample-size", "5", "--seed", "1"])
    assert rc == 0
    assert path.exists()
    assert len(load_eval_set(path)) == 5


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
