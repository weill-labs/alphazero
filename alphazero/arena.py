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
    compute_loss,
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


def opening_temperature_schedule(move_index: int) -> float:
    """Explore early openings, then play greedily from MCTS visits."""

    return 1.0 if move_index < 3 else 0.0


def _self_play_cfg(
    cfg: Mapping[str, object] | None,
    seed: int,
) -> dict[str, object]:
    merged: dict[str, object] = {
        "num_simulations": 64,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25,
    }
    if cfg is not None:
        merged.update(cfg)
    if float(merged.get("dirichlet_eps", 0.0)) <= 0.0:
        merged["dirichlet_eps"] = 0.25
    if float(merged.get("dirichlet_alpha", 0.0)) <= 0.0:
        merged["dirichlet_alpha"] = 0.3
    merged["seed"] = seed
    return merged


def train_tictactoe_agent(
    *,
    iterations: int = 25,
    self_play_games_per_iteration: int = 8,
    self_play_mcts_cfg: Mapping[str, object] | None = None,
    replay_capacity: int = 8192,
    batch_size: int = 64,
    epochs: int = 2,
    lr: float = 5e-3,
    l2_reg: float = 1e-5,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[AlphaZeroNet, dict[str, float | int | str]]:
    """Train a compact tic-tac-toe agent from tabula-rasa self-play only."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if self_play_games_per_iteration <= 0:
        raise ValueError("self_play_games_per_iteration must be positive")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    game = TicTacToe()
    net = AlphaZeroNet(game.num_planes, game.board_shape, game.action_size)
    optimizer = make_optimizer(net, optimizer_name="adam", lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)

    metrics: dict[str, float | int | str] = {}
    for iteration in range(iterations):
        examples: list[SelfPlayExample] = []
        for _ in range(self_play_games_per_iteration):
            game_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            examples.extend(
                play_game(
                    net,
                    game,
                    _self_play_cfg(self_play_mcts_cfg, game_seed),
                    temperature_schedule=opening_temperature_schedule,
                )
            )
        loss_before, _ = compute_loss(net, examples, l2_reg=l2_reg)
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
        loss_after, _ = compute_loss(net, examples, l2_reg=l2_reg)
        metrics["iteration"] = iteration + 1
        metrics["self_play_examples"] = len(examples)
        metrics["loss_before"] = float(loss_before.detach().cpu().item())
        metrics["loss_after"] = float(loss_after.detach().cpu().item())
        metrics["loss_delta"] = float(metrics["loss_before"]) - float(
            metrics["loss_after"]
        )

    if checkpoint_path is not None:
        save_checkpoint(net, checkpoint_path, optimizer=optimizer, metrics=metrics)
        metrics["checkpoint_path"] = str(checkpoint_path)
    return net, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a tic-tac-toe AlphaZero agent."
    )
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-play-games", type=int, default=8)
    parser.add_argument("--self-play-sims", type=int, default=96)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-sims", type=int, default=200)
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
            "dirichlet_eps": args.dirichlet_eps,
        },
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
