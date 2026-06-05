"""Anchored Elo ladder evaluation for AlphaZero checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np

from alphazero.c4_certify import Agent, JaxMCTSAgent
from alphazero.games.connectfour import ConnectFour, ConnectFourState
from jaxzero.train import load_checkpoint

AnchorChoice = Literal["earliest", "random"]
PairingMode = Literal["anchored-sequential", "round-robin"]

DEFAULT_GAMES_PER_PAIRING = 8
DEFAULT_MCTS_SIMS = 100
DEFAULT_FIT_ITERATIONS = 200
DEFAULT_ELO_K = 16.0


class WandbRun(Protocol):
    url: str | None

    def log(self, data: Mapping[str, object], step: int | None = None) -> None: ...

    def finish(self) -> None: ...


@dataclass(frozen=True)
class PairingResult:
    """Aggregate match result from player_a's perspective."""

    player_a: str
    player_b: str
    wins_a: int
    draws: int
    wins_b: int

    @property
    def games(self) -> int:
        return self.wins_a + self.draws + self.wins_b

    @property
    def score_a(self) -> float:
        if self.games <= 0:
            raise ValueError("pairing result has no games")
        return (self.wins_a + 0.5 * self.draws) / self.games

    def as_dict(self) -> dict[str, int | str]:
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "wins_a": self.wins_a,
            "draws": self.draws,
            "wins_b": self.wins_b,
            "games": self.games,
        }


@dataclass(frozen=True)
class EloPoint:
    """One checkpoint's anchored Elo estimate."""

    index: int
    name: str
    elo: float
    path: Path | None = None
    is_anchor: bool = False

    def as_dict(self) -> dict[str, bool | float | int | str | None]:
        return {
            "index": self.index,
            "name": self.name,
            "elo": self.elo,
            "path": str(self.path) if self.path is not None else None,
            "is_anchor": self.is_anchor,
        }


@dataclass(frozen=True)
class EloLadderResult:
    """Full ladder output: raw pairings plus the anchored checkpoint curve."""

    anchor_name: str
    mode: PairingMode
    games_per_pairing: int
    seed: int
    ratings: dict[str, float]
    curve: list[EloPoint]
    pairings: list[PairingResult]

    def as_dict(self) -> dict[str, object]:
        return {
            "anchor_name": self.anchor_name,
            "mode": self.mode,
            "games_per_pairing": self.games_per_pairing,
            "seed": self.seed,
            "ratings": dict(self.ratings),
            "curve": [point.as_dict() for point in self.curve],
            "pairings": [pairing.as_dict() for pairing in self.pairings],
        }


@dataclass(frozen=True)
class LadderContestant:
    """A ladder participant with a seedable player factory."""

    name: str
    player_factory: Callable[[int], Agent]
    path: Path | None = None
    is_checkpoint: bool = True


class RandomAgent:
    """Uniformly samples legal Connect Four moves."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.game = ConnectFour()

    def move(self, state: ConnectFourState) -> int:
        legal_moves = self.game.legal_moves(state)
        if not legal_moves:
            raise ValueError("cannot select a move for a terminal/full state")
        return int(self.rng.choice(legal_moves))

    def value(self, state: ConnectFourState) -> float:
        del state
        return 0.0


def play_match(
    player_a: Agent,
    player_b: Agent,
    game: ConnectFour,
    n_games: int,
) -> tuple[int, int, int]:
    """Play ``n_games`` on Connect Four, alternating seats."""

    if n_games <= 0:
        raise ValueError("n_games must be positive")

    wins_a = 0
    draws = 0
    wins_b = 0
    for game_index in range(n_games):
        state = game.initial_state()
        player_a_mark = 1 if game_index % 2 == 0 else -1

        while not game.is_terminal(state):
            player = (
                player_a if game.current_player(state) == player_a_mark else player_b
            )
            action = int(player.move(state))
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


def fit_elo_ratings(
    names: Sequence[str],
    results: Sequence[PairingResult],
    *,
    anchor_name: str,
    iterations: int = DEFAULT_FIT_ITERATIONS,
    k: float = DEFAULT_ELO_K,
) -> dict[str, float]:
    """Fit Elo ratings from pairwise W/D/L results with a fixed zero anchor."""

    ordered_names = list(names)
    _validate_unique_names(ordered_names)
    if anchor_name not in ordered_names:
        raise ValueError(f"anchor {anchor_name!r} is not in names")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    ratings = {name: 0.0 for name in ordered_names}
    for result in results:
        _validate_pairing_result(result, ratings)

    for _ in range(iterations):
        for result in results:
            expected_a = _expected_score(
                ratings[result.player_a],
                ratings[result.player_b],
            )
            delta = k * (result.score_a - expected_a)
            ratings[result.player_a] += delta
            ratings[result.player_b] -= delta

            anchor_offset = ratings[anchor_name]
            if anchor_offset != 0.0:
                for name in ratings:
                    ratings[name] -= anchor_offset

    ratings[anchor_name] = 0.0
    return {name: float(ratings[name]) for name in ordered_names}


def evaluate_player_ladder(
    contestants: Sequence[LadderContestant],
    game: ConnectFour,
    *,
    anchor_name: str,
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    mode: PairingMode = "anchored-sequential",
    seed: int = 0,
    fit_iterations: int = DEFAULT_FIT_ITERATIONS,
    elo_k: float = DEFAULT_ELO_K,
) -> EloLadderResult:
    """Play deterministic pairings among contestants and fit anchored Elo."""

    if games_per_pairing <= 0:
        raise ValueError("games_per_pairing must be positive")
    contestant_list = list(contestants)
    if not contestant_list:
        raise ValueError("contestants must not be empty")
    contestant_by_name = {contestant.name: contestant for contestant in contestant_list}
    _validate_unique_names([contestant.name for contestant in contestant_list])
    if anchor_name not in contestant_by_name:
        raise ValueError(f"anchor {anchor_name!r} is not a contestant")

    checkpoint_names = [
        contestant.name for contestant in contestant_list if contestant.is_checkpoint
    ]
    pairing_names = _pairing_names(
        [contestant.name for contestant in contestant_list],
        checkpoint_names,
        anchor_name=anchor_name,
        mode=mode,
    )

    rng = np.random.default_rng(seed)
    results: list[PairingResult] = []
    for player_a_name, player_b_name in pairing_names:
        player_a = contestant_by_name[player_a_name].player_factory(
            _next_seed(rng),
        )
        player_b = contestant_by_name[player_b_name].player_factory(
            _next_seed(rng),
        )
        wins_a, draws, wins_b = play_match(
            player_a,
            player_b,
            game,
            games_per_pairing,
        )
        results.append(
            PairingResult(
                player_a=player_a_name,
                player_b=player_b_name,
                wins_a=wins_a,
                draws=draws,
                wins_b=wins_b,
            )
        )

    names = [contestant.name for contestant in contestant_list]
    ratings = fit_elo_ratings(
        names,
        results,
        anchor_name=anchor_name,
        iterations=fit_iterations,
        k=elo_k,
    )
    curve: list[EloPoint] = []
    for contestant in contestant_list:
        if not contestant.is_checkpoint:
            continue
        curve.append(
            EloPoint(
                index=len(curve),
                name=contestant.name,
                elo=ratings[contestant.name],
                path=contestant.path,
                is_anchor=contestant.name == anchor_name,
            )
        )
    return EloLadderResult(
        anchor_name=anchor_name,
        mode=mode,
        games_per_pairing=games_per_pairing,
        seed=seed,
        ratings=ratings,
        curve=curve,
        pairings=results,
    )


def evaluate_checkpoint_ladder(
    game: ConnectFour,
    checkpoint_paths: Sequence[str | Path],
    *,
    anchor: AnchorChoice = "earliest",
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    mode: PairingMode = "anchored-sequential",
    mcts_cfg: Mapping[str, object] | None = None,
    seed: int = 0,
    fit_iterations: int = DEFAULT_FIT_ITERATIONS,
    elo_k: float = DEFAULT_ELO_K,
    wandb_run: WandbRun | None = None,
    wandb_step: int | None = None,
) -> EloLadderResult:
    """Load checkpoints, run seeded MCTS matches, and fit anchored Elo."""

    paths = [Path(path) for path in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must not be empty")
    names = _checkpoint_names(paths)

    contestants = [
        LadderContestant(
            name=name,
            player_factory=_checkpoint_player_factory(path, game, mcts_cfg),
            path=path,
            is_checkpoint=True,
        )
        for name, path in zip(names, paths, strict=True)
    ]
    if anchor == "earliest":
        anchor_name = contestants[0].name
    elif anchor == "random":
        anchor_name = "random"
        contestants = [
            LadderContestant(
                name=anchor_name,
                player_factory=lambda player_seed: RandomAgent(seed=player_seed),
                is_checkpoint=False,
            ),
            *contestants,
        ]
    else:
        raise ValueError("anchor must be 'earliest' or 'random'")

    result = evaluate_player_ladder(
        contestants,
        game,
        anchor_name=anchor_name,
        games_per_pairing=games_per_pairing,
        mode=mode,
        seed=seed,
        fit_iterations=fit_iterations,
        elo_k=elo_k,
    )
    if wandb_run is not None:
        log_elo_curve(wandb_run, result, step=wandb_step)
    return result


def resolve_checkpoint_paths(
    *,
    checkpoints: Sequence[str | Path] = (),
    checkpoint_dir: str | Path | None = None,
    pattern: str = "*.msgpack",
) -> list[Path]:
    """Resolve explicit checkpoint paths plus an optional training-order glob."""

    paths = [Path(path) for path in checkpoints]
    if checkpoint_dir is not None:
        root = Path(checkpoint_dir).expanduser()
        paths.extend(sorted(root.glob(pattern), key=_checkpoint_ladder_sort_key))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser()
        if normalized in seen:
            continue
        unique_paths.append(normalized)
        seen.add(normalized)
    return unique_paths


def _checkpoint_ladder_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"iter_(\d+)\.msgpack", path.name)
    if match is not None:
        return (0, int(match.group(1)), path.name)
    if path.name == "final.msgpack":
        return (1, 0, path.name)
    return (2, 0, path.name)


def log_elo_curve(
    run: WandbRun,
    result: EloLadderResult,
    *,
    step: int | None = None,
) -> None:
    """Log the anchored Elo curve to an existing wandb run."""

    metrics: dict[str, object] = {
        "eval/checkpoint_elo/pairings": len(result.pairings),
        "eval/checkpoint_elo/games_per_pairing": result.games_per_pairing,
        "eval/checkpoint_elo/anchor": result.ratings[result.anchor_name],
    }
    for point in result.curve:
        metrics[f"eval/checkpoint_elo/{point.name}"] = point.elo

    try:
        run.log(metrics, step=step)
    except Exception as exc:
        print(f"Warning: wandb Elo ladder log skipped: {exc}", file=sys.stderr)


def _checkpoint_player_factory(
    checkpoint_path: Path,
    game: ConnectFour,
    mcts_cfg: Mapping[str, object] | None,
) -> Callable[[int], Agent]:
    model = load_checkpoint(checkpoint_path)
    cfg = dict(mcts_cfg or {})

    if not isinstance(game, ConnectFour):
        raise ValueError("JAX Elo ladder currently supports Connect Four only")

    def make_player(player_seed: int) -> Agent:
        return JaxMCTSAgent(
            model,
            sims=int(cfg.get("num_simulations", DEFAULT_MCTS_SIMS)),
            seed=player_seed,
        )

    return make_player


def _checkpoint_names(paths: Sequence[Path]) -> list[str]:
    stems = [path.stem for path in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{index:04d}_{path.stem}" for index, path in enumerate(paths)]


def _pairing_names(
    ordered_names: Sequence[str],
    checkpoint_names: Sequence[str],
    *,
    anchor_name: str,
    mode: PairingMode,
) -> list[tuple[str, str]]:
    order = {name: index for index, name in enumerate(ordered_names)}

    if mode == "round-robin":
        return list(combinations(ordered_names, 2))
    if mode != "anchored-sequential":
        raise ValueError("mode must be 'anchored-sequential' or 'round-robin'")

    pairs: set[tuple[str, str]] = set()
    for checkpoint_name in checkpoint_names:
        if checkpoint_name != anchor_name:
            pairs.add(_ordered_pair(anchor_name, checkpoint_name, order))
    for left, right in zip(checkpoint_names, checkpoint_names[1:], strict=False):
        pairs.add(_ordered_pair(left, right, order))

    return sorted(pairs, key=lambda pair: (order[pair[0]], order[pair[1]]))


def _ordered_pair(
    player_a: str,
    player_b: str,
    order: Mapping[str, int],
) -> tuple[str, str]:
    if player_a == player_b:
        raise ValueError("cannot pair a contestant with itself")
    if order[player_a] < order[player_b]:
        return player_a, player_b
    return player_b, player_a


def _validate_unique_names(names: Sequence[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("contestant names must be unique")


def _validate_pairing_result(
    result: PairingResult,
    ratings: Mapping[str, float],
) -> None:
    if result.player_a not in ratings:
        raise ValueError(f"unknown player {result.player_a!r}")
    if result.player_b not in ratings:
        raise ValueError(f"unknown player {result.player_b!r}")
    if result.player_a == result.player_b:
        raise ValueError("pairing result cannot compare a player with itself")
    if result.wins_a < 0 or result.draws < 0 or result.wins_b < 0:
        raise ValueError("pairing counts must be non-negative")
    if result.games <= 0:
        raise ValueError("pairing result must include at least one game")


def _expected_score(rating: float, opponent_rating: float) -> float:
    exponent = np.clip((opponent_rating - rating) / 400.0, -50.0, 50.0)
    return float(1.0 / (1.0 + 10.0**exponent))


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _init_wandb(
    enabled: bool,
    *,
    project: str,
    run_name: str | None,
    config: Mapping[str, object],
) -> WandbRun | None:
    if not enabled:
        return None
    try:
        wandb_module = __import__("wandb")
        init = getattr(wandb_module, "init")
        return cast(WandbRun, init(project=project, name=run_name, config=dict(config)))
    except Exception as exc:
        print(f"Warning: wandb disabled: {exc}", file=sys.stderr)
        return None


def _finish_wandb(run: WandbRun | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"Warning: wandb finish skipped: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an anchored Elo ladder across AlphaZero checkpoints."
    )
    parser.add_argument("checkpoints", type=Path, nargs="*")
    parser.add_argument("--game", choices=("connectfour",), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*.msgpack")
    parser.add_argument(
        "--anchor",
        choices=("earliest", "random"),
        default="earliest",
    )
    parser.add_argument(
        "--mode",
        choices=("anchored-sequential", "round-robin"),
        default="anchored-sequential",
    )
    parser.add_argument(
        "--games-per-pairing", type=int, default=DEFAULT_GAMES_PER_PAIRING
    )
    parser.add_argument("--sims", type=int, default=DEFAULT_MCTS_SIMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-iterations", type=int, default=DEFAULT_FIT_ITERATIONS)
    parser.add_argument("--elo-k", type=float, default=DEFAULT_ELO_K)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)

    if args.games_per_pairing <= 0:
        parser.error("--games-per-pairing must be positive")
    if args.sims <= 0:
        parser.error("--sims must be positive")
    if args.fit_iterations <= 0:
        parser.error("--fit-iterations must be positive")
    if args.elo_k <= 0:
        parser.error("--elo-k must be positive")

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and not args.checkpoints:
        checkpoint_dir = Path("checkpoints") / args.game
    checkpoint_paths = resolve_checkpoint_paths(
        checkpoints=args.checkpoints,
        checkpoint_dir=checkpoint_dir,
        pattern=args.pattern,
    )
    if not checkpoint_paths:
        parser.error("no checkpoints found")

    game = ConnectFour()
    wandb_project = args.wandb_project or f"alphazero-{args.game}"
    config = {
        "game": args.game,
        "checkpoint_count": len(checkpoint_paths),
        "anchor": args.anchor,
        "mode": args.mode,
        "games_per_pairing": args.games_per_pairing,
        "sims": args.sims,
        "seed": args.seed,
        "fit_iterations": args.fit_iterations,
        "elo_k": args.elo_k,
    }
    wandb_run = _init_wandb(
        args.wandb,
        project=wandb_project,
        run_name=args.wandb_run_name,
        config=config,
    )
    try:
        result = evaluate_checkpoint_ladder(
            game,
            checkpoint_paths,
            anchor=cast(AnchorChoice, args.anchor),
            games_per_pairing=args.games_per_pairing,
            mode=cast(PairingMode, args.mode),
            mcts_cfg={
                "num_simulations": args.sims,
            },
            seed=args.seed,
            fit_iterations=args.fit_iterations,
            elo_k=args.elo_k,
            wandb_run=wandb_run,
        )
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0
    finally:
        _finish_wandb(wandb_run)


if __name__ == "__main__":
    raise SystemExit(main())
