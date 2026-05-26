"""Tests for the AlphaZero benchmark harness."""

from __future__ import annotations

from alphazero.benchmark import BenchmarkConfig, main, run_benchmark


def _tiny_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        game_name="tictactoe",
        self_play_games=1,
        mcts_simulations=1,
        train_epochs=1,
        batch_size=64,
        seed=0,
    )


def test_benchmark_breakdown_has_positive_timings_and_throughputs() -> None:
    result = run_benchmark(_tiny_config())

    assert result.examples > 0
    assert result.network_evals > 0
    assert result.train_steps > 0
    assert result.breakdown.self_play_seconds > 0
    assert result.breakdown.network_inference_seconds > 0
    assert result.breakdown.mcts_non_inference_seconds > 0
    assert result.breakdown.train_step_seconds > 0
    assert result.breakdown.train_iteration_seconds > 0
    assert result.throughput.self_play_games_per_sec > 0
    assert result.throughput.mcts_net_evals_per_sec > 0
    assert result.throughput.network_inference_evals_per_sec > 0
    assert result.throughput.train_steps_per_sec > 0


def test_profile_mode_runs_without_error(capsys) -> None:
    exit_code = main(
        [
            "--game",
            "tictactoe",
            "--self-play-games",
            "1",
            "--mcts-sims",
            "1",
            "--train-epochs",
            "1",
            "--batch-size",
            "64",
            "--profile",
            "--profile-top",
            "5",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "AlphaZero benchmark" in output
    assert "cProfile top 5 by cumulative time" in output
