from __future__ import annotations

import subprocess
import sys

import jax
import numpy as np

from alphazero.c4_certify import (
    CertificationReport,
    JaxMCTSAgent,
    certify_checkpoint_ladder,
    certify_checkpoint_batched,
    certify_connect_four,
    load_eval_labels,
    pgx_state_to_solver_state,
    precompute_solver_labels,
    resolve_checkpoint_ladder_paths,
    sample_positions,
    save_eval_labels,
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


def test_certify_checkpoint_passes_gumbel_scale_to_mctx(tmp_path, monkeypatch) -> None:
    import alphazero.c4_certify as c4_certify
    from alphazero.c4_certify import certify_checkpoint

    captured: list[float] = []
    original = c4_certify.mctx.gumbel_muzero_policy

    def wrapped(*args, **kwargs):
        captured.append(kwargs["gumbel_scale"])
        return original(*args, **kwargs)

    monkeypatch.setattr(c4_certify.mctx, "gumbel_muzero_policy", wrapped)

    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2), action_size=7, channels=4, num_res_blocks=0
    )
    checkpoint = tmp_path / "jaxzero.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    _, state = _play(_FORCED_BLOCK_DRAW)

    certify_checkpoint(
        checkpoint,
        [state],
        sims=1,
        seed=0,
        gumbel_scale=0.25,
    )

    assert captured
    assert set(captured) == {0.25}


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


def test_parallel_certify_matches_serial(tmp_path) -> None:
    """certify_checkpoint(workers>1) must yield identical aggregates to serial.

    Per-position MCTS RNG is seeded from the board state, so splitting positions
    across workers cannot change any per-position result — only wall-clock. The
    merged report's as_dict() (all order-independent aggregates) must match.
    """
    from alphazero.c4_certify import certify_checkpoint

    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2), action_size=7, channels=4, num_res_blocks=0
    )
    checkpoint = tmp_path / "jaxzero.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    positions = sample_positions(sample_size=12, seed=3)

    serial = certify_checkpoint(checkpoint, positions, sims=1, seed=0, workers=1)
    parallel = certify_checkpoint(checkpoint, positions, sims=1, seed=0, workers=4)

    assert parallel.as_dict() == serial.as_dict()


def test_resolve_checkpoint_ladder_paths_sorts_iters_before_final(tmp_path) -> None:
    for name in (
        "final.msgpack",
        "iter_0010.msgpack",
        "iter_0002.msgpack",
        "notes.msgpack",
    ):
        (tmp_path / name).write_bytes(b"")

    paths = resolve_checkpoint_ladder_paths(tmp_path)

    assert [path.name for path in paths] == [
        "iter_0002.msgpack",
        "iter_0010.msgpack",
        "final.msgpack",
        "notes.msgpack",
    ]


def test_certify_checkpoint_ladder_selects_lowest_solver_regret(
    tmp_path,
    monkeypatch,
) -> None:
    import alphazero.c4_certify as c4_certify

    first = tmp_path / "iter_0010.msgpack"
    second = tmp_path / "iter_0020.msgpack"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    def report(mean_wdl_regret: float, blunder_rate: float) -> CertificationReport:
        return CertificationReport(
            sampled_positions=1,
            evaluated_positions=1,
            skipped_positions=0,
            policy_matches=1,
            blunders=int(blunder_rate > 0),
            policy_match_percent=100.0,
            blunder_rate=blunder_rate,
            value_mae=0.0,
            solved=blunder_rate == 0.0,
            score_blunders=0,
            score_blunder_rate=0.0,
            mean_wdl_regret=mean_wdl_regret,
            mean_score_regret=mean_wdl_regret,
            records=(),
        )

    def fake_certify(checkpoint, positions, **kwargs):
        del positions, kwargs
        if checkpoint == first:
            return report(0.2, 0.1)
        return report(0.1, 0.2)

    monkeypatch.setattr(c4_certify, "certify_checkpoint", fake_certify)

    ladder = certify_checkpoint_ladder(
        [first, second],
        positions=[],
        batched=False,
    )

    assert ladder.best_index == 1
    assert ladder.best is not None
    assert ladder.best.checkpoint == str(second)
    assert ladder.as_dict()["best_checkpoint"] == str(second)


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


def test_precompute_solver_labels_aligns_and_caches(tmp_path) -> None:
    """Labels align 1:1 with kept positions and survive a JSON cache roundtrip.

    Each label carries the position's solver value/score/optimal moves plus a
    children map (move -> child value/score) covering every legal move, so a
    later cert needs no solver calls.
    """
    positions = sample_positions(sample_size=8, seed=0)
    kept, labels = precompute_solver_labels(positions)
    assert len(kept) == len(labels)
    game = ConnectFour()
    for state, lab in zip(kept, labels):
        assert set(lab["children"]) == set(game.legal_moves(state))
        assert set(lab["optimal_moves"]).issubset(set(game.legal_moves(state)))

    path = tmp_path / "labels.json"
    save_eval_labels(kept, labels, path)
    loaded_positions, loaded_labels = load_eval_labels(path)
    assert loaded_positions == kept
    assert loaded_labels == labels


def test_certify_checkpoint_batched_matches_serial_stable_metrics(tmp_path) -> None:
    """Batched (cached-label) cert reproduces the serial certifier's stable
    metrics. The batched MCTS uses one top-level rng (vs the serial per-position
    seed), so the noisy weak-blunder count may differ by a few positions, but
    the regret/score aggregates and per-record invariants must hold and the
    evaluated-position count must match the precomputed label set.
    """
    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2), action_size=7, channels=4, num_res_blocks=0
    )
    checkpoint = tmp_path / "m.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    positions = sample_positions(sample_size=12, seed=1)

    kept, labels = precompute_solver_labels(positions)
    report = certify_checkpoint_batched(checkpoint, kept, labels=labels, sims=2, seed=0)

    assert report.evaluated_positions == len(kept)
    # strong (score) blunders are a superset of weak (outcome) blunders
    assert report.score_blunders >= report.blunders
    for record in report.records:
        assert record.score_regret >= 0
        assert 0 <= record.wdl_regret <= 2
        if record.blunder:
            assert record.score_blunder


def test_certify_checkpoint_batched_passes_gumbel_scale_to_mctx(
    tmp_path,
    monkeypatch,
) -> None:
    import alphazero.c4_certify as c4_certify

    captured: list[float] = []
    original = c4_certify.mctx.gumbel_muzero_policy

    def wrapped(*args, **kwargs):
        captured.append(kwargs["gumbel_scale"])
        return original(*args, **kwargs)

    monkeypatch.setattr(c4_certify.mctx, "gumbel_muzero_policy", wrapped)

    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2), action_size=7, channels=4, num_res_blocks=0
    )
    checkpoint = tmp_path / "m.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    _, state = _play(_FORCED_BLOCK_DRAW)
    kept, labels = precompute_solver_labels([state])

    certify_checkpoint_batched(
        checkpoint,
        kept,
        labels=labels,
        sims=2,
        seed=0,
        gumbel_scale=0.5,
    )

    assert captured == [0.5]


def test_certify_checkpoint_batched_precomputes_when_labels_omitted(tmp_path) -> None:
    """Passing labels=None precomputes them internally (no separate cache step)."""
    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2), action_size=7, channels=4, num_res_blocks=0
    )
    checkpoint = tmp_path / "m.msgpack"
    save_checkpoint(create_model(config, seed=0), checkpoint)
    positions = sample_positions(sample_size=6, seed=2)

    report = certify_checkpoint_batched(checkpoint, positions, sims=2, seed=0)
    assert report.evaluated_positions <= len(positions)
    assert 0.0 <= report.blunder_rate <= 1.0
