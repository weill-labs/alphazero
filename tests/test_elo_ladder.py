"""Tests for anchored checkpoint Elo ladder fitting."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from alphazero.elo_ladder import (
    LadderContestant,
    PairingResult,
    evaluate_checkpoint_ladder,
    evaluate_player_ladder,
    fit_elo_ratings,
    play_match,
)
from alphazero.games.connectfour import ConnectFour, ConnectFourState
from jaxzero.net import AlphaZeroNetConfig, create_model
from jaxzero.train import save_checkpoint


class ColumnAgent:
    def __init__(self, column: int) -> None:
        self.column = column
        self.game = ConnectFour()

    def move(self, state: ConnectFourState) -> int:
        if self.column in self.game.legal_moves(state):
            return self.column
        return self.game.legal_moves(state)[0]

    def value(self, state: ConnectFourState) -> float:
        del state
        return 0.0


class SeededRandomAgent:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.game = ConnectFour()

    def move(self, state: ConnectFourState) -> int:
        return int(self.rng.choice(self.game.legal_moves(state)))

    def value(self, state: ConnectFourState) -> float:
        del state
        return 0.0


def test_fit_keeps_anchor_at_zero() -> None:
    ratings = fit_elo_ratings(
        ["anchor", "checkpoint"],
        [PairingResult("anchor", "checkpoint", 1, 2, 1)],
        anchor_name="anchor",
    )

    assert np.isclose(ratings["anchor"], 0.0)


def test_checkpoint_beating_everyone_gets_highest_elo() -> None:
    ratings = fit_elo_ratings(
        ["anchor", "mid", "strong"],
        [
            PairingResult("strong", "anchor", 8, 0, 0),
            PairingResult("strong", "mid", 8, 0, 0),
            PairingResult("mid", "anchor", 4, 0, 4),
        ],
        anchor_name="anchor",
    )

    assert ratings["strong"] > ratings["mid"]
    assert ratings["strong"] > ratings["anchor"]


def test_monotone_win_pattern_yields_monotone_elo() -> None:
    names = ["ckpt0", "ckpt1", "ckpt2", "ckpt3"]
    ratings = fit_elo_ratings(
        names,
        [
            PairingResult("ckpt0", "ckpt1", 0, 0, 8),
            PairingResult("ckpt1", "ckpt2", 0, 0, 8),
            PairingResult("ckpt2", "ckpt3", 0, 0, 8),
        ],
        anchor_name="ckpt0",
    )

    assert [ratings[name] for name in names] == sorted(ratings[name] for name in names)
    assert ratings["ckpt0"] < ratings["ckpt1"] < ratings["ckpt2"] < ratings["ckpt3"]


def test_player_ladder_is_deterministic_for_fixed_seed() -> None:
    contestants = [
        LadderContestant("anchor", lambda seed: SeededRandomAgent(seed)),
        LadderContestant("candidate", lambda seed: SeededRandomAgent(seed)),
        LadderContestant("latest", lambda seed: SeededRandomAgent(seed)),
    ]

    first = evaluate_player_ladder(
        contestants,
        ConnectFour(),
        anchor_name="anchor",
        games_per_pairing=2,
        mode="round-robin",
        seed=123,
    )
    second = evaluate_player_ladder(
        contestants,
        ConnectFour(),
        anchor_name="anchor",
        games_per_pairing=2,
        mode="round-robin",
        seed=123,
    )

    assert first.as_dict() == second.as_dict()


def test_play_match_uses_agent_move_on_connect_four() -> None:
    wins_a, draws, wins_b = play_match(
        ColumnAgent(0),
        ColumnAgent(1),
        ConnectFour(),
        n_games=1,
    )

    assert (wins_a, draws, wins_b) == (1, 0, 0)


def test_checkpoint_ladder_loads_jax_agents(tmp_path) -> None:
    game = ConnectFour()
    config = AlphaZeroNetConfig(
        obs_shape=(6, 7, 2),
        action_size=game.action_size,
        channels=4,
        num_res_blocks=0,
    )
    early = tmp_path / "early.msgpack"
    late = tmp_path / "late.msgpack"
    save_checkpoint(create_model(config, seed=0), early)
    save_checkpoint(create_model(config, seed=1), late)

    result = evaluate_checkpoint_ladder(
        game,
        [early, late],
        games_per_pairing=1,
        mcts_cfg={"num_simulations": 1},
        seed=7,
        fit_iterations=1,
    )

    assert [point.name for point in result.curve] == ["early", "late"]
    assert [point.path for point in result.curve] == [early, late]
    assert len(result.pairings) == 1
    assert result.pairings[0].games == 1


def test_elo_ladder_imports_do_not_load_torch() -> None:
    code = "import sys, alphazero.elo_ladder; raise SystemExit('torch' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
