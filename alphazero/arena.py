"""Evaluation players and match runner for AlphaZero agents."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from alphazero.game import Game, State
from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet
from alphazero.selfplay import SelfPlayExample, play_game
from alphazero.train import (
    ReplayBuffer,
    make_optimizer,
    save_checkpoint,
    train_iteration,
)


class Player(Protocol):
    def select_action(self, game: Game, state: State) -> int: ...


class RandomPlayer:
    """Uniformly samples legal moves."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game, state: State) -> int:
        legal = game.legal_moves(state)
        if not legal:
            raise ValueError("cannot select an action for a terminal state")
        return int(self.rng.choice(legal))


class PerfectPlayer:
    """Minimax player for small deterministic zero-sum games such as tic-tac-toe."""

    def __init__(self) -> None:
        self._value_cache: dict[tuple[int, State], int] = {}

    def select_action(self, game: Game, state: State) -> int:
        legal = game.legal_moves(state)
        if not legal:
            raise ValueError("cannot select an action for a terminal state")

        scores = {
            action: -self.value(game, game.apply_move(state, action))
            for action in legal
        }
        best_score = max(scores.values())
        return min(action for action, score in scores.items() if score == best_score)

    def policy_value(self, game: Game, state: State) -> tuple[np.ndarray, int]:
        """Return a uniform optimal policy and minimax value for `state`."""

        legal = game.legal_moves(state)
        pi = np.zeros(game.action_size, dtype=np.float32)
        value = self.value(game, state)
        if not legal:
            return pi, value

        scores = {
            action: -self.value(game, game.apply_move(state, action))
            for action in legal
        }
        best_score = max(scores.values())
        best_actions = [
            action for action, score in scores.items() if score == best_score
        ]
        for action in best_actions:
            pi[action] = 1.0 / len(best_actions)
        return pi, value

    def value(self, game: Game, state: State) -> int:
        """Minimax value from the current player's perspective."""

        key = (id(game), state)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached

        winner = game.winner(state)
        if winner is not None:
            if winner == 0:
                value = 0
            elif winner == game.current_player(state):
                value = 1
            else:
                value = -1
            self._value_cache[key] = value
            return value

        value = max(
            -self.value(game, game.apply_move(state, action))
            for action in game.legal_moves(state)
        )
        self._value_cache[key] = value
        return value


class MCTSPlayer:
    """MCTS policy player backed by a network with the AlphaZero `predict` API."""

    def __init__(
        self,
        net,
        *,
        c_puct: float = 1.5,
        num_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.0,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game, state: State) -> int:
        mcts = MCTS(
            self.net,
            game,
            c_puct=self.c_puct,
            num_simulations=self.num_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_eps=self.dirichlet_eps,
            seed=int(self.rng.integers(0, np.iinfo(np.int32).max)),
        )
        pi = mcts.run(state, add_noise=False)
        if pi.sum() <= 0:
            legal = game.legal_moves(state)
            if not legal:
                raise ValueError("cannot select an action for a terminal state")
            return int(self.rng.choice(legal))
        return mcts.select_action(pi, temperature=self.temperature, rng=self.rng)


def play_match(
    player_a: Player, player_b: Player, game: Game, n_games: int
) -> tuple[int, int, int]:
    """Play `n_games`, alternating seats, and return (wins_a, draws, wins_b)."""

    if n_games <= 0:
        raise ValueError("n_games must be positive")

    wins_a = 0
    draws = 0
    wins_b = 0
    for game_index in range(n_games):
        state = game.initial_state()
        player_a_mark = 1 if game_index % 2 == 0 else -1

        while not game.is_terminal(state):
            current_player = game.current_player(state)
            player = player_a if current_player == player_a_mark else player_b
            action = player.select_action(game, state)
            if action not in game.legal_moves(state):
                raise ValueError(f"player selected illegal action {action}")
            state = game.apply_move(state, action)

        winner = game.winner(state)
        if winner == 0:
            draws += 1
        elif winner == player_a_mark:
            wins_a += 1
        else:
            wins_b += 1

    return wins_a, draws, wins_b


def perfect_training_examples(game: Game) -> list[SelfPlayExample]:
    """Enumerate reachable non-terminal states with minimax policy/value targets."""

    player = PerfectPlayer()
    examples: list[SelfPlayExample] = []
    visited: set[State] = set()

    def visit(state: State) -> None:
        if state in visited:
            return
        visited.add(state)
        if game.is_terminal(state):
            return

        pi, value = player.policy_value(game, state)
        examples.append((game.encode(state).copy(), pi, value))
        for action in game.legal_moves(state):
            visit(game.apply_move(state, action))

    visit(game.initial_state())
    return examples


def train_tictactoe_agent(
    *,
    iterations: int = 2,
    self_play_games_per_iteration: int = 0,
    self_play_mcts_cfg: Mapping[str, object] | None = None,
    perfect_examples_limit: int | None = 256,
    replay_capacity: int = 4096,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 1e-2,
    l2_reg: float = 1e-5,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[AlphaZeroNet, dict[str, float | int | str]]:
    """Train a compact tic-tac-toe agent using minimax bootstrap plus optional self-play."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if perfect_examples_limit is not None and perfect_examples_limit <= 0:
        raise ValueError("perfect_examples_limit must be positive when provided")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    game = TicTacToe()
    net = AlphaZeroNet(game.num_planes, game.board_shape, game.action_size)
    optimizer = make_optimizer(net, optimizer_name="adam", lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)
    perfect_examples = perfect_training_examples(game)
    if perfect_examples_limit is not None and perfect_examples_limit < len(
        perfect_examples
    ):
        indices = rng.choice(
            len(perfect_examples), size=perfect_examples_limit, replace=False
        )
        perfect_examples = [perfect_examples[int(i)] for i in indices]

    metrics: dict[str, float | int | str] = {}
    for iteration in range(iterations):
        examples = list(perfect_examples)
        for _ in range(self_play_games_per_iteration):
            examples.extend(
                play_game(
                    net,
                    game,
                    self_play_mcts_cfg
                    or {"num_simulations": 16, "dirichlet_eps": 0.0, "seed": seed},
                    temperature_schedule=0.0,
                )
            )
        metrics = train_iteration(
            net,
            examples,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            epochs=epochs,
            l2_reg=l2_reg,
            shuffle=True,
            rng=rng,
        )
        metrics["iteration"] = iteration + 1

    if checkpoint_path is not None:
        save_checkpoint(net, checkpoint_path, optimizer=optimizer, metrics=metrics)
        metrics["checkpoint_path"] = str(checkpoint_path)
    return net, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a tic-tac-toe AlphaZero agent."
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-play-games", type=int, default=0)
    parser.add_argument("--self-play-sims", type=int, default=32)
    parser.add_argument("--perfect-examples-limit", type=int, default=256)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/tictactoe.pt")
    )
    args = parser.parse_args(argv)

    net, metrics = train_tictactoe_agent(
        iterations=args.iterations,
        self_play_games_per_iteration=args.self_play_games,
        self_play_mcts_cfg={
            "num_simulations": args.self_play_sims,
            "dirichlet_eps": 0.0,
            "seed": args.seed,
        },
        perfect_examples_limit=args.perfect_examples_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )
    wins, draws, losses = play_match(
        MCTSPlayer(net, num_simulations=args.eval_sims, seed=args.seed),
        PerfectPlayer(),
        TicTacToe(),
        args.eval_games,
    )
    print(
        {
            "metrics": metrics,
            "vs_perfect": {"wins": wins, "draws": draws, "losses": losses},
        }
    )
    return 0 if losses == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
