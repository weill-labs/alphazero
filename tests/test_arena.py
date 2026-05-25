"""Tests for arena evaluation and tic-tac-toe verification players."""

from __future__ import annotations

from alphazero.arena import MCTSPlayer, PerfectPlayer, play_match, train_tictactoe_agent
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


def test_short_training_run_never_loses_to_perfect_player(tmp_path) -> None:
    game = TicTacToe()
    net, metrics = train_tictactoe_agent(
        iterations=1,
        self_play_games_per_iteration=0,
        perfect_examples_limit=256,
        batch_size=128,
        epochs=4,
        checkpoint_path=tmp_path / "tictactoe.pt",
        seed=0,
    )
    trained = MCTSPlayer(net, num_simulations=64, seed=0)

    wins, draws, losses = play_match(trained, PerfectPlayer(), game, n_games=2)

    assert metrics["num_examples"] >= 256
    assert wins + draws + losses == 2
    assert losses == 0
