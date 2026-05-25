"""Tests for arena evaluation and tic-tac-toe verification players."""

from __future__ import annotations

import builtins

from alphazero.arena import (
    MCTSPlayer,
    PerfectPlayer,
    RandomPlayer,
    play_match,
    train_tictactoe_agent,
)
from alphazero.game import Game, State
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


def test_wandb_disabled_by_default_does_not_import_wandb(monkeypatch, tmp_path) -> None:
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
    assert wins + draws + losses == 2


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
