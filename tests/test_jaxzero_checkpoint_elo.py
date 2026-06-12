"""Tests for pgx checkpoint Elo ladders."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from jaxzero.checkpoint_elo import (
    PairingResult,
    evaluate_checkpoint_ladder,
    evaluate_checkpoint_stability,
    evaluate_fixed_position_search,
    fit_elo_ratings,
    main,
    resolve_checkpoint_paths,
)
from jaxzero.net import AlphaZeroNetConfig, create_model
from jaxzero.selfplay import initial_observation_shape, make_env
from jaxzero.train import save_checkpoint


def _save_checkpoint(path, *, game: str = "othello", seed: int = 0) -> None:
    env = make_env(game)
    config = AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(game),
        action_size=env.num_actions,
        channels=4,
        num_res_blocks=0,
    )
    save_checkpoint(create_model(config, seed=seed), path)


def test_resolve_checkpoint_paths_orders_periodic_before_final(tmp_path) -> None:
    final = tmp_path / "final.msgpack"
    iter_2 = tmp_path / "iter_0002.msgpack"
    iter_1 = tmp_path / "iter_0001.msgpack"
    other = tmp_path / "other.msgpack"
    for path in (final, iter_2, iter_1, other):
        path.touch()

    paths = resolve_checkpoint_paths(checkpoint_dir=tmp_path)

    assert paths == [iter_1, iter_2, final, other]


def test_othello_checkpoint_ladder_loads_and_matches(tmp_path) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    result = evaluate_checkpoint_ladder(
        [early, late],
        game="othello",
        games_per_pairing=2,
        max_steps=128,
        seed=7,
        fit_iterations=1,
    )

    assert result.game == "othello"
    assert result.evaluator_mode == "greedy"
    assert [point.name for point in result.curve] == ["early", "late"]
    assert [point.path for point in result.curve] == [early, late]
    assert len(result.pairings) == 1
    assert result.pairings[0].games == 2
    assert result.max_steps == 128
    assert result.as_dict()["best_checkpoint"] in {str(early), str(late)}


def test_othello_mcts_checkpoint_ladder_loads_and_matches(tmp_path) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    result = evaluate_checkpoint_ladder(
        [early, late],
        game="othello",
        evaluator_mode="mcts",
        mcts_simulations=1,
        gumbel_scale=0.0,
        games_per_pairing=2,
        max_steps=2,
        seed=7,
        fit_iterations=1,
    )

    assert result.game == "othello"
    assert result.evaluator_mode == "mcts"
    assert result.mcts_simulations == 1
    assert result.gumbel_scale == 0.0
    assert len(result.pairings) == 1
    assert result.pairings[0].games == 2


def test_checkpoint_stability_sweeps_budgets_and_seeds(tmp_path) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    result = evaluate_checkpoint_stability(
        [early, late],
        game="othello",
        games_per_pairing=2,
        max_steps=2,
        mcts_simulations_list=[1, 2],
        seeds=[3, 4],
        fit_iterations=1,
        instability_threshold=0.1,
    )

    assert result["game"] == "othello"
    assert result["evaluator_mode"] == "mcts"
    assert result["mcts_simulations"] == [1, 2]
    assert result["seeds"] == [3, 4]
    assert len(result["runs"]) == 4
    assert set(result["rating_summary"]) == {"early", "late"}
    assert result["pairing_summary"][0]["player_a"] == "early"
    assert result["pairing_summary"][0]["player_b"] == "late"
    assert len(result["pairing_summary"][0]["runs"]) == 4


def test_fixed_position_search_scores_shared_position_batch(tmp_path) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    result = evaluate_fixed_position_search(
        [early, late],
        game="othello",
        num_positions=4,
        min_ply=0,
        max_ply=1,
        max_steps=3,
        mcts_simulations_list=[1, 2],
        seeds=[3, 4],
        position_seed=5,
    )

    assert result["game"] == "othello"
    assert result["evaluator_mode"] == "fixed-position-mcts"
    assert result["num_positions"] == 4
    assert result["mcts_simulations"] == [1, 2]
    assert result["seeds"] == [3, 4]
    assert result["reference_checkpoint"] == "early"
    assert set(result["checkpoint_summary"]) == {"early", "late"}
    assert len(result["runs"]) == 8
    assert result["position_summary"]["reached_positions"] == 4

    early_summary = result["checkpoint_summary"]["early"]
    assert 0.0 <= early_summary["action_stability"] <= 1.0
    assert 0.0 <= early_summary["consensus_match"] <= 1.0
    assert 0.0 <= early_summary["reference_match"] <= 1.0
    assert "majority_action_counts" in early_summary


def test_checkpoint_ladder_is_deterministic_for_fixed_seed(tmp_path) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    first = evaluate_checkpoint_ladder(
        [early, late],
        game="othello",
        games_per_pairing=2,
        max_steps=2,
        seed=123,
        fit_iterations=1,
    )
    second = evaluate_checkpoint_ladder(
        [early, late],
        game="othello",
        games_per_pairing=2,
        max_steps=2,
        seed=123,
        fit_iterations=1,
    )

    assert first.as_dict() == second.as_dict()


def test_games_per_pairing_must_be_even_for_balanced_seats(tmp_path) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    with pytest.raises(ValueError, match="balance seats"):
        evaluate_checkpoint_ladder(
            [early, late],
            game="othello",
            games_per_pairing=3,
            max_steps=2,
        )


def test_fit_elo_ratings_ranks_monotone_results() -> None:
    ratings = fit_elo_ratings(
        ["early", "mid", "late"],
        [
            PairingResult("early", "mid", 0, 0, 8),
            PairingResult("mid", "late", 0, 0, 8),
        ],
        anchor_name="early",
    )

    assert ratings["early"] < ratings["mid"] < ratings["late"]


def test_othello_ladder_rejects_c4_checkpoint_shape(tmp_path) -> None:
    c4_checkpoint = tmp_path / "c4.msgpack"
    other = tmp_path / "other.msgpack"
    _save_checkpoint(c4_checkpoint, game="connectfour", seed=0)
    _save_checkpoint(other, seed=1)

    with pytest.raises(ValueError, match="does not match game 'othello'"):
        evaluate_checkpoint_ladder(
            [c4_checkpoint, other],
            game="othello",
            games_per_pairing=2,
            max_steps=2,
        )


def test_checkpoint_elo_cli_prints_json(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "2",
            "--fit-iterations",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["game"] == "othello"
    assert payload["evaluator_mode"] == "greedy"
    assert payload["pairings"][0]["games"] == 2


def test_checkpoint_elo_cli_accepts_mcts_mode(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--evaluator-mode",
            "mcts",
            "--mcts-simulations",
            "1",
            "--gumbel-scale",
            "0.0",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "2",
            "--fit-iterations",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["evaluator_mode"] == "mcts"
    assert payload["mcts_simulations"] == 1
    assert payload["gumbel_scale"] == 0.0


def test_checkpoint_elo_cli_traces_batched_match(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--evaluator-mode",
            "mcts",
            "--mcts-simulations",
            "1",
            "--gumbel-scale",
            "0.0",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "2",
            "--trace-plies",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["game"] == "othello"
    assert payload["evaluator_mode"] == "mcts"
    assert payload["mcts_simulations"] == 1
    assert payload["pairing"]["games"] == 2
    assert len(payload["summaries"]) == 2
    assert payload["summaries"][0]["active_lanes"] == 2
    assert payload["summaries"][0]["selected_action_counts"]
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["ply"] == 0
    assert len(payload["steps"][0]["lanes"]) == 2
    assert {
        "action",
        "action_a",
        "action_b",
        "active",
        "actor",
        "current_player",
        "lane",
        "player_a_seat",
        "return_a",
        "reward_a",
    } <= set(payload["steps"][0]["lanes"][0])


def test_checkpoint_elo_cli_can_trace_summaries_only(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--evaluator-mode",
            "mcts",
            "--mcts-simulations",
            "1",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "2",
            "--trace-plies",
            "2",
            "--trace-summary-only",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["pairing"]["games"] == 2
    assert len(payload["summaries"]) == 2
    assert "steps" not in payload


def test_checkpoint_elo_cli_probes_match_state(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--mcts-simulations",
            "1",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "2",
            "--probe-ply",
            "1",
            "--probe-budgets",
            "1,2",
            "--probe-top-k",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["game"] == "othello"
    assert payload["target_ply"] == 1
    assert payload["replay_simulations"] == 1
    assert payload["probe_simulations"] == [1, 2]
    assert len(payload["summaries"]) == 2
    assert len(payload["lanes"]) == 2
    assert len(payload["lanes"][0]["budgets"]) == 2
    assert payload["lanes"][0]["budgets"][0]["top_actions"]


def test_checkpoint_elo_cli_evaluates_forced_actions(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--mcts-simulations",
            "1",
            "--continuation-simulations",
            "1",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "3",
            "--force-ply",
            "1",
            "--force-actions",
            "0,1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["game"] == "othello"
    assert payload["target_ply"] == 1
    assert payload["replay_simulations"] == 1
    assert payload["continuation_simulations"] == 1
    assert payload["force_actor_role"] == "player_a"
    assert [entry["action"] for entry in payload["actions"]] == [0, 1]
    assert "forced_result" in payload["actions"][0]
    assert "target_default_action_counts" in payload["actions"][0]


def test_checkpoint_elo_cli_runs_stability_sweep(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--games-per-pairing",
            "2",
            "--max-steps",
            "2",
            "--fit-iterations",
            "1",
            "--stability-budgets",
            "1,2",
            "--stability-seeds",
            "3,4",
            "--stability-score-threshold",
            "0.1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["evaluator_mode"] == "mcts"
    assert payload["mcts_simulations"] == [1, 2]
    assert payload["seeds"] == [3, 4]
    assert len(payload["runs"]) == 4
    assert payload["pairing_summary"][0]["runs"][0]["games"] == 2


def test_checkpoint_elo_cli_runs_fixed_position_search(tmp_path, capsys) -> None:
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    _save_checkpoint(early, seed=0)
    _save_checkpoint(late, seed=1)

    exit_code = main(
        [
            str(early),
            str(late),
            "--game",
            "othello",
            "--max-steps",
            "3",
            "--mcts-simulations",
            "1",
            "--position-samples",
            "2",
            "--position-min-ply",
            "0",
            "--position-max-ply",
            "1",
            "--position-budgets",
            "1,2",
            "--position-seeds",
            "3,4",
            "--position-seed",
            "5",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["evaluator_mode"] == "fixed-position-mcts"
    assert payload["num_positions"] == 2
    assert payload["mcts_simulations"] == [1, 2]
    assert payload["seeds"] == [3, 4]
    assert set(payload["checkpoint_summary"]) == {"early", "late"}


def test_checkpoint_elo_imports_without_c4_solver_dependency() -> None:
    code = (
        "import sys, jaxzero.checkpoint_elo; "
        "raise SystemExit("
        "'alphazero.c4_certify' in sys.modules or "
        "'alphazero.games.connectfour' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
