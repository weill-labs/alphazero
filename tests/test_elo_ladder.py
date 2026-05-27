"""Tests for anchored checkpoint Elo ladder fitting."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from alphazero.elo_ladder import (
    LadderContestant,
    PairingResult,
    evaluate_player_ladder,
    fit_elo_ratings,
)
from alphazero.game import Game, State


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


class SeededRandomActionPlayer:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game, state: State) -> int:
        return int(self.rng.choice(game.legal_moves(state)))


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
        LadderContestant("anchor", lambda seed: SeededRandomActionPlayer(seed)),
        LadderContestant("candidate", lambda seed: SeededRandomActionPlayer(seed)),
        LadderContestant("latest", lambda seed: SeededRandomActionPlayer(seed)),
    ]

    first = evaluate_player_ladder(
        contestants,
        OneMoveGame(),
        anchor_name="anchor",
        games_per_pairing=6,
        mode="round-robin",
        seed=123,
    )
    second = evaluate_player_ladder(
        contestants,
        OneMoveGame(),
        anchor_name="anchor",
        games_per_pairing=6,
        mode="round-robin",
        seed=123,
    )

    assert first.as_dict() == second.as_dict()
