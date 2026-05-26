"""Command-line human-vs-agent play for trained AlphaZero checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from alphazero.arena import MCTSPlayer
from alphazero.game import Game, State
from alphazero.games.connectfour import ConnectFour
from alphazero.games.tictactoe import TicTacToe
from alphazero.network import AlphaZeroNet


class Player(Protocol):
    def select_action(self, game: Game, state: State) -> int: ...


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass(frozen=True)
class PlayResult:
    winner: int
    human_player: int
    message: str


def game_from_name(name: str) -> Game:
    if name == "tictactoe":
        return TicTacToe()
    if name == "connectfour":
        return ConnectFour()
    raise ValueError("game must be 'tictactoe' or 'connectfour'")


def parse_move(raw_move: str) -> int:
    text = raw_move.strip()
    if not text:
        raise ValueError("enter a move")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("enter an integer move") from exc


def validate_move(game: Game, state: State, action: int) -> int:
    legal = game.legal_moves(state)
    if action not in legal:
        legal_text = ", ".join(str(move) for move in legal)
        raise ValueError(f"move {action} is not legal; legal moves: {legal_text}")
    return action


def parse_legal_move(raw_move: str, game: Game, state: State) -> int:
    return validate_move(game, state, parse_move(raw_move))


def read_human_move(
    game: Game,
    state: State,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    while True:
        legal_text = ", ".join(str(move) for move in game.legal_moves(state))
        raw_move = input_fn(f"Your move ({legal_text}): ")
        try:
            return parse_legal_move(raw_move, game, state)
        except ValueError as exc:
            output_fn(f"Invalid move: {exc}")


def outcome_message(winner: int, human_player: int) -> str:
    if winner == 0:
        return "Draw."
    if winner == human_player:
        return "You win."
    return "You lose."


def play_human_vs_agent(
    game: Game,
    agent: Player,
    *,
    human_first: bool = False,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> PlayResult:
    state = game.initial_state()
    human_player = 1 if human_first else -1
    output_fn(game.__str__(state))

    while not game.is_terminal(state):
        if game.current_player(state) == human_player:
            action = read_human_move(
                game,
                state,
                input_fn=input_fn,
                output_fn=output_fn,
            )
        else:
            action = agent.select_action(game, state)
            validate_move(game, state, action)
            output_fn(f"Agent plays {action}.")

        state = game.apply_move(state, action)
        output_fn(game.__str__(state))

    winner = game.winner(state)
    if winner is None:
        raise RuntimeError("terminal state did not report a winner")
    message = outcome_message(winner, human_player)
    output_fn(message)
    return PlayResult(winner=winner, human_player=human_player, message=message)


def load_checkpoint(path: str | Path, game: Game) -> AlphaZeroNet:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
    else:
        model_state = checkpoint
    if not isinstance(model_state, Mapping):
        raise ValueError(f"{checkpoint_path} does not contain a model state")

    net = AlphaZeroNet(game.num_planes, game.board_shape, game.action_size)
    net.load_state_dict(model_state)
    net.train(False)
    return net


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    parser = argparse.ArgumentParser(
        description="Play a human game against a trained AlphaZero checkpoint."
    )
    parser.add_argument("--game", choices=("tictactoe", "connectfour"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--human-first", action="store_true")
    args = parser.parse_args(argv)

    if args.sims <= 0:
        parser.error("--sims must be positive")

    game = game_from_name(args.game)
    net = load_checkpoint(args.checkpoint, game)
    agent = MCTSPlayer(net, num_simulations=args.sims)
    play_human_vs_agent(
        game,
        agent,
        human_first=args.human_first,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
