"""Tests for arena evaluation and tic-tac-toe verification players."""

from __future__ import annotations

import builtins

from alphazero.arena import (
    MCTSPlayer,
    PerfectPlayer,
    RandomPlayer,
    evaluate_connect_four_tactics,
    immediate_blocking_moves,
    immediate_winning_moves,
    play_match,
    train_agent,
    train_tictactoe_agent,
)
from alphazero.game import Game, State
from alphazero.games.connectfour import ConnectFour
from alphazero.games.tictactoe import TicTacToe


class RowPlayer:
    def __init__(self, preferred: tuple[int, ...]) -> None:
        self.preferred = preferred

    def select_action(self, game: Game, state: State) -> int:
        legal = game.legal_moves(state)
        for action in self.preferred:
            if action in legal:
                return action
        return legal[0]


class TacticalOraclePlayer:
    def select_action(self, game: Game, state: State) -> int:
        for targets in (
            immediate_winning_moves(game, state),
            immediate_blocking_moves(game, state),
        ):
            if targets:
                return targets[0]
        return game.legal_moves(state)[0]


class TacticalAvoiderPlayer:
    def select_action(self, game: Game, state: State) -> int:
        targets = immediate_winning_moves(game, state)
        if not targets:
            targets = immediate_blocking_moves(game, state)
        for action in game.legal_moves(state):
            if action not in targets:
                return action
        return game.legal_moves(state)[0]


def test_perfect_player_vs_perfect_player_always_draws() -> None:
    game = TicTacToe()

    wins_a, draws, wins_b = play_match(
        PerfectPlayer(), PerfectPlayer(), game, n_games=4
    )

    assert (wins_a, draws, wins_b) == (0, 4, 0)


def test_play_match_tallies_wins_draws_and_losses() -> None:
    game = TicTacToe()
    top_row = RowPlayer((0, 1, 2))
    bottom_row = RowPlayer((6, 7, 8))

    wins_a, draws, wins_b = play_match(top_row, bottom_row, game, n_games=1)

    assert (wins_a, draws, wins_b) == (1, 0, 0)


def test_short_self_play_training_beats_random_and_reduces_loss(tmp_path) -> None:
    game = TicTacToe()
    net, metrics = train_tictactoe_agent(
        iterations=2,
        self_play_games_per_iteration=4,
        self_play_mcts_cfg={
            "num_simulations": 24,
            "dirichlet_eps": 0.25,
            "seed": 0,
        },
        batch_size=32,
        epochs=2,
        checkpoint_path=tmp_path / "tictactoe.pt",
        seed=0,
    )
    trained = MCTSPlayer(net, num_simulations=48, seed=0)

    wins, draws, losses = play_match(trained, RandomPlayer(seed=1), game, n_games=12)

    assert metrics["self_play_examples"] > 0
    assert metrics["loss_after"] < metrics["loss_before"]
    assert wins + draws + losses == 12
    assert wins >= 8
    assert wins > losses


def test_train_wandb_disabled_by_default_does_not_import_wandb(
    monkeypatch, tmp_path
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("wandb should not be imported unless enabled")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    game = TicTacToe()

    net, metrics = train_tictactoe_agent(
        iterations=1,
        self_play_games_per_iteration=1,
        self_play_mcts_cfg={
            "num_simulations": 8,
            "dirichlet_eps": 0.25,
            "seed": 1,
        },
        batch_size=8,
        epochs=1,
        checkpoint_path=tmp_path / "tictactoe.pt",
        seed=1,
    )
    wins, draws, losses = play_match(
        MCTSPlayer(net, num_simulations=8, seed=1),
        RandomPlayer(seed=2),
        game,
        n_games=2,
    )

    assert metrics["self_play_examples"] > 0
    assert metrics["iteration_seconds"] > 0
    assert metrics["iters_per_sec"] > 0
    assert metrics["self_play_games_per_sec"] > 0
    assert wins + draws + losses == 2


def test_train_agent_connect_four_one_iteration_yields_net(tmp_path) -> None:
    game = ConnectFour()

    net, metrics = train_agent(
        game,
        iterations=1,
        self_play_games_per_iteration=1,
        self_play_mcts_cfg={
            "num_simulations": 2,
            "dirichlet_eps": 0.25,
            "seed": 3,
        },
        batch_size=16,
        epochs=1,
        checkpoint_path=tmp_path / "connectfour.pt",
        seed=3,
    )

    assert net.action_size == game.action_size
    assert net.board_shape == game.board_shape
    assert metrics["self_play_examples"] > 0
    assert metrics["checkpoint_path"] == str(tmp_path / "connectfour.pt")


def test_connect_four_tactical_metrics_score_oracle_and_avoider() -> None:
    game = ConnectFour()

    oracle_metrics = evaluate_connect_four_tactics(TacticalOraclePlayer(), game)
    avoider_metrics = evaluate_connect_four_tactics(TacticalAvoiderPlayer(), game)

    expected_oracle_metrics = {
        "immediate_win_rate": 1.0,
        "block_rate": 1.0,
    }
    if oracle_metrics != expected_oracle_metrics:
        raise AssertionError(oracle_metrics)
    if avoider_metrics["immediate_win_rate"] >= 1.0:
        raise AssertionError(avoider_metrics)
    if avoider_metrics["block_rate"] >= 1.0:
        raise AssertionError(avoider_metrics)


def test_wandb_import_failure_is_nonfatal(monkeypatch, tmp_path, capsys) -> None:
    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "wandb" or name.startswith("wandb."):
            raise RuntimeError("wandb unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    _, metrics = train_tictactoe_agent(
        iterations=1,
        self_play_games_per_iteration=1,
        self_play_mcts_cfg={
            "num_simulations": 4,
            "dirichlet_eps": 0.25,
            "seed": 2,
        },
        batch_size=8,
        epochs=1,
        checkpoint_path=tmp_path / "tictactoe.pt",
        seed=2,
        wandb_enabled=True,
    )

    assert metrics["self_play_examples"] > 0
    assert "Warning: wandb disabled" in capsys.readouterr().err
