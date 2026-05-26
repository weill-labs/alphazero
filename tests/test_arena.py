"""Tests for arena evaluation and tic-tac-toe verification players."""

from __future__ import annotations

import builtins
from typing import NamedTuple

import numpy as np
import torch

import alphazero.arena as arena
from alphazero.arena import (
    DEFAULT_ELO,
    DEFAULT_LADDER_DEPTHS,
    MCTSPlayer,
    PerfectPlayer,
    RandomPlayer,
    evaluate_connect_four_tactics,
    evaluate_ladder,
    gating_match,
    immediate_blocking_moves,
    immediate_winning_moves,
    play_match,
    train_agent,
    train_tictactoe_agent,
    update_elo,
)
from alphazero.game import Game, State
from alphazero.games.connectfour import ConnectFour
from alphazero.games.tictactoe import TicTacToe


class OneMoveState(NamedTuple):
    player: int
    winner: int | None = None


class OneMoveGame(Game):
    action_size = 2
    board_shape = (1, 2)
    num_planes = 2

    def initial_state(self) -> OneMoveState:
        return OneMoveState(player=1)

    def current_player(self, s: OneMoveState) -> int:
        return s.player

    def legal_moves(self, s: OneMoveState) -> list[int]:
        return [] if self.is_terminal(s) else [0, 1]

    def apply_move(self, s: OneMoveState, a: int) -> OneMoveState:
        winner = s.player if a == 1 else -s.player
        return OneMoveState(player=-s.player, winner=winner)

    def is_terminal(self, s: OneMoveState) -> bool:
        return s.winner is not None

    def winner(self, s: OneMoveState) -> int | None:
        return s.winner

    def encode(self, s: OneMoveState) -> np.ndarray:
        return np.zeros((self.num_planes, *self.board_shape), dtype=np.float32)

    def __str__(self, s: OneMoveState) -> str:
        return str(s)


class FixedActionPlayer:
    def __init__(self, action: int) -> None:
        self.action = action

    def select_action(self, game: Game, state: State) -> int:
        if self.action not in game.legal_moves(state):
            raise ValueError(f"fixed action {self.action} is illegal")
        return self.action


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


def _stub_self_play_examples(game: Game):
    policy = np.ones(game.action_size, dtype=np.float32) / game.action_size
    return [(game.encode(game.initial_state()), policy, 0.0)]


def _stub_loss(*args, **kwargs):
    return torch.tensor(1.0), {}


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


def test_gating_promotes_clearly_stronger_candidate() -> None:
    result = gating_match(
        FixedActionPlayer(1),
        FixedActionPlayer(0),
        OneMoveGame(),
        n_games=4,
        threshold=0.55,
    )

    assert result["promoted"] == 1
    assert result["winrate"] == 1.0


def test_gating_rejects_weaker_candidate() -> None:
    result = gating_match(
        FixedActionPlayer(0),
        FixedActionPlayer(1),
        OneMoveGame(),
        n_games=4,
        threshold=0.55,
    )

    assert result["promoted"] == 0
    assert result["winrate"] == 0.0


def test_elo_moves_up_after_win_and_down_after_loss() -> None:
    base_rating = 1000.0

    assert update_elo(base_rating, base_rating, 1.0) > base_rating
    assert update_elo(base_rating, base_rating, 0.0) < base_rating


def test_ladder_eval_returns_expected_winrate_keys() -> None:
    metrics = evaluate_ladder(
        FixedActionPlayer(1),
        OneMoveGame(),
        n_games=2,
        seed=0,
    )

    assert DEFAULT_LADDER_DEPTHS == [1, 2, 4, 6]
    assert set(metrics) == {
        "eval/ladder_random_winrate",
        *{f"eval/ladder_negamax_d{depth}_winrate" for depth in DEFAULT_LADDER_DEPTHS},
    }
    for winrate in metrics.values():
        assert 0.0 <= winrate <= 1.0


def test_train_agent_self_play_uses_latest_net(monkeypatch) -> None:
    game = OneMoveGame()
    seen_weight_sums: list[float] = []

    def fake_play_game(net, game_arg, cfg, temperature_schedule):
        seen_weight_sums.append(float(next(net.parameters()).detach().sum().item()))
        return _stub_self_play_examples(game_arg)

    def fake_train_iteration(net, examples, **kwargs):
        with torch.no_grad():
            next(net.parameters()).add_(1.0)
        return {"loss": 1.0}

    monkeypatch.setattr(arena, "play_game", fake_play_game)
    monkeypatch.setattr(arena, "compute_loss", _stub_loss)
    monkeypatch.setattr(arena, "train_iteration", fake_train_iteration)

    train_agent(
        game,
        iterations=2,
        self_play_games_per_iteration=1,
        batch_size=1,
        epochs=1,
        gating_interval=99,
        eval_interval=99,
        seed=0,
    )

    assert len(seen_weight_sums) == 2
    assert seen_weight_sums[1] > seen_weight_sums[0]


def test_train_agent_returns_and_checkpoints_latest_net_when_gate_rejects(
    monkeypatch, tmp_path
) -> None:
    game = OneMoveGame()
    saved: dict[str, object] = {}

    def fake_play_game(net, game_arg, cfg, temperature_schedule):
        return _stub_self_play_examples(game_arg)

    def fake_train_iteration(net, examples, **kwargs):
        with torch.no_grad():
            next(net.parameters()).fill_(3.0)
        return {"loss": 1.0}

    def fake_gating_match(candidate, best, game_arg, *, n_games, threshold):
        return {
            "wins": 0,
            "draws": 0,
            "losses": n_games,
            "winrate": 0.0,
            "score": 0.0,
            "promoted": 0,
        }

    def fake_save_checkpoint(net, checkpoint_path, *, optimizer=None, metrics=None):
        saved["path"] = checkpoint_path
        saved["weight_sum"] = float(next(net.parameters()).detach().sum().item())
        saved["has_optimizer"] = optimizer is not None

    monkeypatch.setattr(arena, "play_game", fake_play_game)
    monkeypatch.setattr(arena, "compute_loss", _stub_loss)
    monkeypatch.setattr(arena, "train_iteration", fake_train_iteration)
    monkeypatch.setattr(arena, "gating_match", fake_gating_match)
    monkeypatch.setattr(arena, "save_checkpoint", fake_save_checkpoint)

    net, metrics = train_agent(
        game,
        iterations=1,
        self_play_games_per_iteration=1,
        batch_size=1,
        epochs=1,
        checkpoint_path=tmp_path / "model.pt",
        gating_interval=1,
        gating_games=2,
        eval_interval=99,
        seed=0,
    )

    returned_sum = float(next(net.parameters()).detach().sum().item())
    assert returned_sum == saved["weight_sum"]
    assert returned_sum > 0.0
    assert saved["has_optimizer"]
    assert metrics["checkpoint_path"] == str(tmp_path / "model.pt")


def test_train_agent_gating_promotes_candidate_against_reference_and_bumps_elo(
    monkeypatch,
) -> None:
    game = OneMoveGame()

    def fake_play_game(net, game_arg, cfg, temperature_schedule):
        return _stub_self_play_examples(game_arg)

    def fake_train_iteration(net, examples, **kwargs):
        with torch.no_grad():
            next(net.parameters()).fill_(42.0)
        return {"loss": 1.0}

    def fake_mcts_player(net, cfg, *, seed):
        first_weight = float(next(net.parameters()).flatten()[0].detach().item())
        return FixedActionPlayer(1 if first_weight > 10.0 else 0)

    monkeypatch.setattr(arena, "play_game", fake_play_game)
    monkeypatch.setattr(arena, "compute_loss", _stub_loss)
    monkeypatch.setattr(arena, "train_iteration", fake_train_iteration)
    monkeypatch.setattr(arena, "_mcts_player", fake_mcts_player)

    _, metrics = train_agent(
        game,
        iterations=1,
        self_play_games_per_iteration=1,
        batch_size=1,
        epochs=1,
        gating_interval=1,
        gating_games=4,
        gating_threshold=0.55,
        eval_interval=99,
        seed=0,
    )

    assert metrics["eval/promoted"] == 1
    assert metrics["eval/gating_winrate"] == 1.0
    assert metrics["eval/elo"] > DEFAULT_ELO


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


def test_arena_main_wandb_project_defaults_to_game_and_allows_override(
    monkeypatch, capsys
) -> None:
    captured: list[tuple[bool, str, str, str]] = []

    def fake_init_wandb(enabled, *, project, run_name, config):
        captured.append(
            (
                enabled,
                project,
                str(config["wandb_project"]),
                str(config["game"]),
            )
        )
        return None

    def fake_train_agent(game, **kwargs):
        return object(), {"self_play_examples": 1}

    monkeypatch.setattr(arena, "_init_wandb", fake_init_wandb)
    monkeypatch.setattr(arena, "train_agent", fake_train_agent)
    monkeypatch.setattr(
        arena,
        "evaluate_connect_four_tactics",
        lambda *args, **kwargs: {"immediate_win_rate": 1.0, "block_rate": 1.0},
    )
    monkeypatch.setattr(
        arena,
        "evaluate_connect_four_solver_anchor",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(arena, "play_match", lambda *args, **kwargs: (1, 0, 0))

    assert arena.main(["--game", "connectfour", "--iterations", "1"]) == 0
    assert (
        arena.main(
            [
                "--game",
                "connectfour",
                "--iterations",
                "1",
                "--wandb-project",
                "custom-project",
            ]
        )
        == 0
    )

    assert captured == [
        (True, "alphazero-connectfour", "alphazero-connectfour", "connectfour"),
        (True, "custom-project", "custom-project", "connectfour"),
    ]
    capsys.readouterr()


def test_self_play_cfg_passes_batch_size_through() -> None:
    assert arena._self_play_cfg({"batch_size": 8}, seed=3)["batch_size"] == 8
    # Without an explicit batch size, self-play falls back to MCTS's default.
    assert "batch_size" not in arena._self_play_cfg(None, seed=3)


def test_arena_main_threads_mcts_batch_size_into_self_play(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_train_agent(game, **kwargs):
        captured.update(kwargs)
        return object(), {"self_play_examples": 1}

    monkeypatch.setattr(arena, "_init_wandb", lambda *a, **k: None)
    monkeypatch.setattr(arena, "train_agent", fake_train_agent)
    monkeypatch.setattr(arena, "play_match", lambda *a, **k: (1, 0, 0))
    monkeypatch.setattr(
        arena,
        "evaluate_connect_four_tactics",
        lambda *a, **k: {"immediate_win_rate": 1.0, "block_rate": 1.0},
    )
    monkeypatch.setattr(
        arena, "evaluate_connect_four_solver_anchor", lambda *a, **k: {}
    )

    assert (
        arena.main(
            ["--game", "connectfour", "--iterations", "1", "--mcts-batch-size", "7"]
        )
        == 0
    )
    assert captured["self_play_mcts_cfg"]["batch_size"] == 7

    captured.clear()
    assert arena.main(["--game", "connectfour", "--iterations", "1"]) == 0
    assert captured["self_play_mcts_cfg"]["batch_size"] == 16  # default


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
