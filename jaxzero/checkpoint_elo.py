"""Checkpoint Elo ladders for pgx-backed jaxzero games."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from flax import nnx

from jaxzero.game_specs import DEFAULT_GAME, resolve_game, supported_games
from jaxzero.net import Net, apply_model
from jaxzero.selfplay import initial_observation_shape, make_env
from jaxzero.train import load_checkpoint

PairingMode = Literal["anchored-sequential", "round-robin"]
EvaluatorMode = Literal["greedy", "mcts"]

DEFAULT_GAMES_PER_PAIRING = 8
DEFAULT_FIT_ITERATIONS = 200
DEFAULT_ELO_K = 16.0
DEFAULT_EVALUATOR_MODE: EvaluatorMode = "greedy"
DEFAULT_MCTS_SIMULATIONS = 32
DEFAULT_GUMBEL_SCALE = 0.0


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

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "wins_a": self.wins_a,
            "draws": self.draws,
            "wins_b": self.wins_b,
            "games": self.games,
            "score_a": self.score_a,
        }


@dataclass(frozen=True)
class EloPoint:
    """One checkpoint's anchored Elo estimate."""

    index: int
    name: str
    elo: float
    path: Path
    is_anchor: bool = False

    def as_dict(self) -> dict[str, bool | float | int | str]:
        return {
            "index": self.index,
            "name": self.name,
            "elo": self.elo,
            "path": str(self.path),
            "is_anchor": self.is_anchor,
        }


@dataclass(frozen=True)
class CheckpointEloResult:
    """Full checkpoint ladder output."""

    game: str
    evaluator_mode: EvaluatorMode
    anchor_name: str
    mode: PairingMode
    games_per_pairing: int
    max_steps: int
    seed: int
    mcts_simulations: int | None
    gumbel_scale: float | None
    ratings: dict[str, float]
    curve: list[EloPoint]
    pairings: list[PairingResult]

    @property
    def best_point(self) -> EloPoint:
        if not self.curve:
            raise ValueError("result has no checkpoint curve")
        return max(self.curve, key=lambda point: point.elo)

    def as_dict(self) -> dict[str, object]:
        best = self.best_point
        return {
            "game": self.game,
            "evaluator_mode": self.evaluator_mode,
            "anchor_name": self.anchor_name,
            "mode": self.mode,
            "games_per_pairing": self.games_per_pairing,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "mcts_simulations": self.mcts_simulations,
            "gumbel_scale": self.gumbel_scale,
            "ratings": dict(self.ratings),
            "curve": [point.as_dict() for point in self.curve],
            "pairings": [pairing.as_dict() for pairing in self.pairings],
            "best_name": best.name,
            "best_elo": best.elo,
            "best_checkpoint": str(best.path),
        }


@dataclass(frozen=True)
class _LoadedCheckpoint:
    name: str
    path: Path
    graphdef: nnx.GraphDef[Net]
    params: nnx.State


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


def resolve_checkpoint_paths(
    *,
    checkpoints: Sequence[str | Path] = (),
    checkpoint_dir: str | Path | None = None,
    pattern: str = "*.msgpack",
) -> list[Path]:
    """Resolve explicit checkpoint paths plus an optional training-order glob."""

    paths = [Path(path).expanduser() for path in checkpoints]
    if checkpoint_dir is not None:
        root = Path(checkpoint_dir).expanduser()
        paths.extend(sorted(root.glob(pattern), key=_checkpoint_ladder_sort_key))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


def evaluate_checkpoint_ladder(
    checkpoint_paths: Sequence[str | Path],
    *,
    game: str = DEFAULT_GAME,
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    max_steps: int | None = None,
    mode: PairingMode = "anchored-sequential",
    evaluator_mode: EvaluatorMode = DEFAULT_EVALUATOR_MODE,
    mcts_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    seed: int = 0,
    fit_iterations: int = DEFAULT_FIT_ITERATIONS,
    elo_k: float = DEFAULT_ELO_K,
) -> CheckpointEloResult:
    """Load checkpoints, run pgx matches, and fit anchored Elo."""

    spec = resolve_game(game)
    if evaluator_mode not in ("greedy", "mcts"):
        raise ValueError("evaluator_mode must be 'greedy' or 'mcts'")
    if games_per_pairing <= 0:
        raise ValueError("games_per_pairing must be positive")
    if games_per_pairing % 2 != 0:
        raise ValueError("games_per_pairing must be even to balance seats")
    if mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if fit_iterations <= 0:
        raise ValueError("fit_iterations must be positive")
    if elo_k <= 0:
        raise ValueError("elo_k must be positive")

    paths = [Path(path) for path in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must not be empty")
    max_steps = spec.default_max_steps if max_steps is None else max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    names = _checkpoint_names(paths)
    checkpoints = [
        _load_checkpoint(path, name=name, game=spec.name)
        for name, path in zip(names, paths, strict=True)
    ]
    anchor_name = checkpoints[0].name
    pairings = _pairing_names(
        [checkpoint.name for checkpoint in checkpoints],
        anchor_name=anchor_name,
        mode=mode,
    )

    rng = np.random.default_rng(seed)
    results: list[PairingResult] = []
    checkpoint_by_name = {checkpoint.name: checkpoint for checkpoint in checkpoints}
    for player_a_name, player_b_name in pairings:
        results.append(
            play_checkpoint_match(
                checkpoint_by_name[player_a_name],
                checkpoint_by_name[player_b_name],
                game=spec.name,
                games=games_per_pairing,
                max_steps=max_steps,
                evaluator_mode=evaluator_mode,
                mcts_simulations=mcts_simulations,
                gumbel_scale=gumbel_scale,
                seed=_next_seed(rng),
            )
        )

    ratings = fit_elo_ratings(
        [checkpoint.name for checkpoint in checkpoints],
        results,
        anchor_name=anchor_name,
        iterations=fit_iterations,
        k=elo_k,
    )
    curve = [
        EloPoint(
            index=index,
            name=checkpoint.name,
            elo=ratings[checkpoint.name],
            path=checkpoint.path,
            is_anchor=checkpoint.name == anchor_name,
        )
        for index, checkpoint in enumerate(checkpoints)
    ]
    return CheckpointEloResult(
        game=spec.name,
        evaluator_mode=evaluator_mode,
        anchor_name=anchor_name,
        mode=mode,
        games_per_pairing=games_per_pairing,
        max_steps=max_steps,
        seed=seed,
        mcts_simulations=mcts_simulations if evaluator_mode == "mcts" else None,
        gumbel_scale=float(gumbel_scale) if evaluator_mode == "mcts" else None,
        ratings=ratings,
        curve=curve,
        pairings=results,
    )


def play_checkpoint_match(
    player_a: _LoadedCheckpoint,
    player_b: _LoadedCheckpoint,
    *,
    game: str,
    games: int,
    max_steps: int,
    seed: int,
    evaluator_mode: EvaluatorMode = DEFAULT_EVALUATOR_MODE,
    mcts_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
) -> PairingResult:
    """Play a balanced match between two loaded checkpoints."""

    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if evaluator_mode not in ("greedy", "mcts"):
        raise ValueError("evaluator_mode must be 'greedy' or 'mcts'")
    if mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")

    if evaluator_mode == "greedy":
        play = _make_greedy_match(
            player_a.graphdef,
            player_b.graphdef,
            game=game,
            num_games=games,
            max_steps=max_steps,
        )
    else:
        play = _make_mcts_match(
            player_a.graphdef,
            player_b.graphdef,
            game=game,
            num_games=games,
            max_steps=max_steps,
            num_simulations=mcts_simulations,
            gumbel_scale=gumbel_scale,
        )
    counts = play(player_a.params, player_b.params, jax.random.PRNGKey(seed))
    wins_a, draws, wins_b = (int(v) for v in jax.device_get(counts).tolist())
    return PairingResult(
        player_a=player_a.name,
        player_b=player_b.name,
        wins_a=wins_a,
        draws=draws,
        wins_b=wins_b,
    )


def trace_checkpoint_game(
    checkpoint_paths: Sequence[str | Path],
    *,
    game: str = DEFAULT_GAME,
    games: int = DEFAULT_GAMES_PER_PAIRING,
    max_steps: int | None = None,
    evaluator_mode: EvaluatorMode = DEFAULT_EVALUATOR_MODE,
    mcts_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    seed: int = 0,
    trace_plies: int = 8,
    summary_only: bool = False,
) -> dict[str, object]:
    """Trace the first plies of a batched checkpoint match."""

    spec = resolve_game(game)
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) != 2:
        raise ValueError("trace requires exactly two checkpoint paths")
    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if trace_plies <= 0:
        raise ValueError("trace_plies must be positive")
    if evaluator_mode not in ("greedy", "mcts"):
        raise ValueError("evaluator_mode must be 'greedy' or 'mcts'")
    if mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")

    max_steps = spec.default_max_steps if max_steps is None else max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    names = _checkpoint_names(paths)
    player_a, player_b = (
        _load_checkpoint(path, name=name, game=spec.name)
        for name, path in zip(names, paths, strict=True)
    )
    return trace_checkpoint_match(
        player_a,
        player_b,
        game=spec.name,
        games=games,
        max_steps=max_steps,
        evaluator_mode=evaluator_mode,
        mcts_simulations=mcts_simulations,
        gumbel_scale=gumbel_scale,
        seed=seed,
        trace_plies=trace_plies,
        summary_only=summary_only,
    )


def trace_checkpoint_match(
    player_a: _LoadedCheckpoint,
    player_b: _LoadedCheckpoint,
    *,
    game: str,
    games: int,
    max_steps: int,
    seed: int,
    evaluator_mode: EvaluatorMode = DEFAULT_EVALUATOR_MODE,
    mcts_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    trace_plies: int = 8,
    summary_only: bool = False,
) -> dict[str, object]:
    """Trace the exact batched match used by ``play_checkpoint_match``."""

    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if trace_plies <= 0:
        raise ValueError("trace_plies must be positive")
    if evaluator_mode not in ("greedy", "mcts"):
        raise ValueError("evaluator_mode must be 'greedy' or 'mcts'")
    if mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")

    trace = _make_match_trace(
        player_a.graphdef,
        player_b.graphdef,
        game=game,
        num_games=games,
        max_steps=max_steps,
        evaluator_mode=evaluator_mode,
        num_simulations=mcts_simulations,
        gumbel_scale=gumbel_scale,
    )
    counts, records = trace(
        player_a.params,
        player_b.params,
        jax.random.PRNGKey(seed),
    )
    counts = [int(v) for v in jax.device_get(counts).tolist()]
    records = jax.device_get(records)
    limit = min(trace_plies, max_steps)
    player_a_seat = records["player_a_seat"][0]
    steps = []
    summaries = []
    for ply in range(limit):
        lanes = []
        selected_counts: dict[str, int] = {}
        selected_by_actor: dict[str, dict[str, int]] = {}
        action_a_counts: dict[str, int] = {}
        action_b_counts: dict[str, int] = {}
        for lane in range(games):
            current_player = int(records["current_player"][ply][lane])
            seat = int(player_a_seat[lane])
            actor = player_a.name if current_player == seat else player_b.name
            active = bool(records["active"][ply][lane])
            action = int(records["action"][ply][lane])
            action_a = int(records["action_a"][ply][lane])
            action_b = int(records["action_b"][ply][lane])
            if active:
                _increment_count(selected_counts, action)
                _increment_nested_count(selected_by_actor, actor, action)
                _increment_count(action_a_counts, action_a)
                _increment_count(action_b_counts, action_b)
            lanes.append(
                {
                    "lane": lane,
                    "active": active,
                    "player_a_seat": seat,
                    "current_player": current_player,
                    "actor": actor,
                    "action_a": action_a,
                    "action_b": action_b,
                    "action": action,
                    "reward_a": float(records["reward_a"][ply][lane]),
                    "return_a": float(records["return_a"][ply][lane]),
                }
            )
        summaries.append(
            {
                "ply": ply,
                "active_lanes": int(np.sum(records["active"][ply])),
                "selected_action_counts": selected_counts,
                "selected_by_actor": selected_by_actor,
                f"{player_a.name}_search_action_counts": action_a_counts,
                f"{player_b.name}_search_action_counts": action_b_counts,
            }
        )
        if not summary_only:
            steps.append({"ply": ply, "lanes": lanes})

    payload = {
        "game": resolve_game(game).name,
        "evaluator_mode": evaluator_mode,
        "mcts_simulations": mcts_simulations if evaluator_mode == "mcts" else None,
        "gumbel_scale": float(gumbel_scale) if evaluator_mode == "mcts" else None,
        "games": games,
        "max_steps": max_steps,
        "seed": seed,
        "trace_plies": limit,
        "player_a": player_a.name,
        "player_b": player_b.name,
        "pairing": PairingResult(
            player_a.name,
            player_b.name,
            wins_a=counts[0],
            draws=counts[1],
            wins_b=counts[2],
        ).as_dict(),
        "summaries": summaries,
    }
    if not summary_only:
        payload["steps"] = steps
    return payload


def probe_checkpoint_state(
    checkpoint_paths: Sequence[str | Path],
    *,
    game: str = DEFAULT_GAME,
    games: int = DEFAULT_GAMES_PER_PAIRING,
    max_steps: int | None = None,
    replay_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    probe_simulations: Sequence[int] = (DEFAULT_MCTS_SIMULATIONS,),
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    seed: int = 0,
    target_ply: int = 0,
    top_k: int = 5,
) -> dict[str, object]:
    """Probe MCTS policy choices at an exact batched-match ply."""

    spec = resolve_game(game)
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) != 2:
        raise ValueError("probe requires exactly two checkpoint paths")
    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if replay_simulations <= 0:
        raise ValueError("replay_simulations must be positive")
    probe_budgets = _validate_probe_simulations(probe_simulations)
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")

    max_steps = spec.default_max_steps if max_steps is None else max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if target_ply < 0 or target_ply >= max_steps:
        raise ValueError("target_ply must be in [0, max_steps)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    names = _checkpoint_names(paths)
    player_a, player_b = (
        _load_checkpoint(path, name=name, game=spec.name)
        for name, path in zip(names, paths, strict=True)
    )
    return probe_checkpoint_match(
        player_a,
        player_b,
        game=spec.name,
        games=games,
        max_steps=max_steps,
        replay_simulations=replay_simulations,
        probe_simulations=probe_budgets,
        gumbel_scale=gumbel_scale,
        seed=seed,
        target_ply=target_ply,
        top_k=top_k,
    )


def probe_checkpoint_match(
    player_a: _LoadedCheckpoint,
    player_b: _LoadedCheckpoint,
    *,
    game: str,
    games: int,
    max_steps: int,
    replay_simulations: int,
    probe_simulations: Sequence[int],
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    seed: int = 0,
    target_ply: int = 0,
    top_k: int = 5,
) -> dict[str, object]:
    """Replay a match to ``target_ply`` and probe policy choices there."""

    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if replay_simulations <= 0:
        raise ValueError("replay_simulations must be positive")
    probe_budgets = _validate_probe_simulations(probe_simulations)
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if target_ply < 0 or target_ply >= max_steps:
        raise ValueError("target_ply must be in [0, max_steps)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    replay = _make_match_state_at_ply(
        player_a.graphdef,
        player_b.graphdef,
        game=game,
        num_games=games,
        max_steps=max_steps,
        target_ply=target_ply,
        replay_simulations=replay_simulations,
        gumbel_scale=gumbel_scale,
    )
    state, return_a, target_key, player_a_seat = replay(
        player_a.params,
        player_b.params,
        jax.random.PRNGKey(seed),
    )
    key_a, key_b = jax.random.split(target_key)

    budget_outputs = {}
    for budget in probe_budgets:
        search_a = _make_mcts_search_details(
            player_a.graphdef,
            game=game,
            num_simulations=budget,
            gumbel_scale=gumbel_scale,
        )
        search_b = _make_mcts_search_details(
            player_b.graphdef,
            game=game,
            num_simulations=budget,
            gumbel_scale=gumbel_scale,
        )
        action_a, weights_a, value_a = search_a(player_a.params, state, key_a)
        action_b, weights_b, value_b = search_b(player_b.params, state, key_b)
        budget_outputs[budget] = {
            "action_a": jax.device_get(action_a),
            "action_b": jax.device_get(action_b),
            "weights_a": jax.device_get(weights_a),
            "weights_b": jax.device_get(weights_b),
            "value_a": jax.device_get(value_a),
            "value_b": jax.device_get(value_b),
        }

    current_player = np.asarray(jax.device_get(state.current_player))
    legal_action_mask = np.asarray(jax.device_get(state.legal_action_mask))
    active = np.asarray(jax.device_get(~(state.terminated | state.truncated)))
    player_a_seat = np.asarray(jax.device_get(player_a_seat))
    return_a = np.asarray(jax.device_get(return_a))

    summaries: list[dict[str, object]] = []
    for budget in probe_budgets:
        output = budget_outputs[budget]
        selected_by_actor: dict[str, dict[str, int]] = {}
        for lane in range(games):
            actor = (
                player_a.name
                if int(current_player[lane]) == int(player_a_seat[lane])
                else player_b.name
            )
            action = (
                int(output["action_a"][lane])
                if actor == player_a.name
                else int(output["action_b"][lane])
            )
            if bool(active[lane]):
                _increment_nested_count(selected_by_actor, actor, action)
        summaries.append(
            {
                "simulations": budget,
                "selected_by_actor": selected_by_actor,
            }
        )

    lanes = []
    for lane in range(games):
        actor = (
            player_a.name
            if int(current_player[lane]) == int(player_a_seat[lane])
            else player_b.name
        )
        budget_entries = []
        for budget in probe_budgets:
            output = budget_outputs[budget]
            if actor == player_a.name:
                action = int(output["action_a"][lane])
                weights = np.asarray(output["weights_a"][lane])
                value = float(np.asarray(output["value_a"])[lane])
            else:
                action = int(output["action_b"][lane])
                weights = np.asarray(output["weights_b"][lane])
                value = float(np.asarray(output["value_b"])[lane])
            budget_entries.append(
                {
                    "simulations": budget,
                    "action": action,
                    "root_value": value,
                    "top_actions": _top_action_weights(
                        weights,
                        legal_action_mask[lane],
                        top_k=top_k,
                    ),
                }
            )
        lanes.append(
            {
                "lane": lane,
                "active": bool(active[lane]),
                "player_a_seat": int(player_a_seat[lane]),
                "current_player": int(current_player[lane]),
                "actor": actor,
                "return_a_before_ply": float(return_a[lane]),
                "legal_actions": [
                    int(action)
                    for action in np.flatnonzero(legal_action_mask[lane]).tolist()
                ],
                "budgets": budget_entries,
            }
        )

    return {
        "game": resolve_game(game).name,
        "games": games,
        "max_steps": max_steps,
        "seed": seed,
        "target_ply": target_ply,
        "replay_simulations": replay_simulations,
        "probe_simulations": probe_budgets,
        "gumbel_scale": float(gumbel_scale),
        "player_a": player_a.name,
        "player_b": player_b.name,
        "summaries": summaries,
        "lanes": lanes,
    }


def _validate_probe_simulations(probe_simulations: Sequence[int]) -> list[int]:
    budgets = [int(value) for value in probe_simulations]
    if not budgets:
        raise ValueError("probe_simulations must not be empty")
    if any(value <= 0 for value in budgets):
        raise ValueError("probe_simulations values must be positive")
    return budgets


def _top_action_weights(
    weights: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, float | int]]:
    legal_actions = np.flatnonzero(legal_action_mask)
    if legal_actions.size == 0:
        return []
    legal_weights = weights[legal_actions]
    order = np.argsort(-legal_weights, kind="stable")[:top_k]
    return [
        {
            "action": int(legal_actions[index]),
            "weight": float(legal_weights[index]),
        }
        for index in order
    ]


def _increment_count(counts: dict[str, int], key: int) -> None:
    counts[str(key)] = counts.get(str(key), 0) + 1


def _increment_nested_count(
    counts: dict[str, dict[str, int]],
    outer_key: str,
    inner_key: int,
) -> None:
    inner = counts.setdefault(outer_key, {})
    _increment_count(inner, inner_key)


def _make_greedy_match(
    graphdef_a: nnx.GraphDef[Net],
    graphdef_b: nnx.GraphDef[Net],
    *,
    game: str,
    num_games: int,
    max_steps: int,
):
    env = pgx.make(resolve_game(game).env_id)
    player_a_seat = jnp.concatenate(
        [
            jnp.zeros(num_games // 2, dtype=jnp.int32),
            jnp.ones(num_games // 2, dtype=jnp.int32),
        ]
    )
    game_index = jnp.arange(num_games)
    neg_inf = jnp.finfo(jnp.float32).min

    @jax.jit
    def play(params_a: nnx.State, params_b: nnx.State, rng_key: jax.Array) -> jax.Array:
        def step(carry, _):
            state, return_a = carry
            logits_a, _ = apply_model(graphdef_a, params_a, state.observation)
            logits_b, _ = apply_model(graphdef_b, params_b, state.observation)
            action_a = jnp.argmax(
                jnp.where(state.legal_action_mask, logits_a, neg_inf), axis=-1
            )
            action_b = jnp.argmax(
                jnp.where(state.legal_action_mask, logits_b, neg_inf), axis=-1
            )
            action = jnp.where(
                state.current_player == player_a_seat,
                action_a,
                action_b,
            )
            state = jax.vmap(env.step)(state, action)
            return_a = return_a + state.rewards[game_index, player_a_seat]
            return (state, return_a), None

        state = jax.vmap(env.init)(jax.random.split(rng_key, num_games))
        (_, return_a), _ = jax.lax.scan(
            step,
            (state, jnp.zeros(num_games)),
            xs=None,
            length=max_steps,
        )
        wins_a = jnp.sum(return_a == 1.0).astype(jnp.int32)
        draws = jnp.sum(return_a == 0.0).astype(jnp.int32)
        wins_b = jnp.sum(return_a == -1.0).astype(jnp.int32)
        return jnp.stack([wins_a, draws, wins_b])

    return play


def _make_match_trace(
    graphdef_a: nnx.GraphDef[Net],
    graphdef_b: nnx.GraphDef[Net],
    *,
    game: str,
    num_games: int,
    max_steps: int,
    evaluator_mode: EvaluatorMode,
    num_simulations: int,
    gumbel_scale: float,
):
    env = pgx.make(resolve_game(game).env_id)
    player_a_seat = jnp.concatenate(
        [
            jnp.zeros(num_games // 2, dtype=jnp.int32),
            jnp.ones(num_games // 2, dtype=jnp.int32),
        ]
    )
    game_index = jnp.arange(num_games)
    neg_inf = jnp.finfo(jnp.float32).min
    if evaluator_mode == "mcts":
        search_a = _make_mcts_search(
            graphdef_a,
            game=game,
            num_simulations=num_simulations,
            gumbel_scale=gumbel_scale,
        )
        search_b = _make_mcts_search(
            graphdef_b,
            game=game,
            num_simulations=num_simulations,
            gumbel_scale=gumbel_scale,
        )
    else:
        search_a = None
        search_b = None

    @jax.jit
    def trace(
        params_a: nnx.State,
        params_b: nnx.State,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        def step(carry, key):
            state, return_a = carry
            active = ~(state.terminated | state.truncated)
            current_player = state.current_player
            if evaluator_mode == "mcts":
                key_a, key_b = jax.random.split(key)
                search_state = _replace_inactive_lanes(state, active, dummy_state)
                action_a = search_a(params_a, search_state, key_a)
                action_b = search_b(params_b, search_state, key_b)
            else:
                logits_a, _ = apply_model(graphdef_a, params_a, state.observation)
                logits_b, _ = apply_model(graphdef_b, params_b, state.observation)
                action_a = jnp.argmax(
                    jnp.where(state.legal_action_mask, logits_a, neg_inf),
                    axis=-1,
                )
                action_b = jnp.argmax(
                    jnp.where(state.legal_action_mask, logits_b, neg_inf),
                    axis=-1,
                )
            action = jnp.where(
                state.current_player == player_a_seat,
                action_a,
                action_b,
            )
            stepped_state = jax.vmap(env.step)(state, action)
            state = _replace_inactive_lanes(stepped_state, active, state)
            step_return = stepped_state.rewards[game_index, player_a_seat]
            return_a = return_a + jnp.where(active, step_return, 0.0)
            record = {
                "active": active,
                "current_player": current_player,
                "player_a_seat": player_a_seat,
                "action_a": action_a,
                "action_b": action_b,
                "action": action,
                "reward_a": jnp.where(active, step_return, 0.0),
                "return_a": return_a,
            }
            return (state, return_a), record

        rng_key, init_key, scan_key = jax.random.split(rng_key, 3)
        state = jax.vmap(env.init)(jax.random.split(init_key, num_games))
        dummy_state = jax.vmap(env.init)(
            jax.random.split(jax.random.PRNGKey(0), num_games)
        )
        (_, return_a), records = jax.lax.scan(
            step,
            (state, jnp.zeros(num_games)),
            xs=jax.random.split(scan_key, max_steps),
        )
        wins_a = jnp.sum(return_a == 1.0).astype(jnp.int32)
        draws = jnp.sum(return_a == 0.0).astype(jnp.int32)
        wins_b = jnp.sum(return_a == -1.0).astype(jnp.int32)
        return jnp.stack([wins_a, draws, wins_b]), records

    return trace


def _make_match_state_at_ply(
    graphdef_a: nnx.GraphDef[Net],
    graphdef_b: nnx.GraphDef[Net],
    *,
    game: str,
    num_games: int,
    max_steps: int,
    target_ply: int,
    replay_simulations: int,
    gumbel_scale: float,
):
    env = pgx.make(resolve_game(game).env_id)
    player_a_seat = jnp.concatenate(
        [
            jnp.zeros(num_games // 2, dtype=jnp.int32),
            jnp.ones(num_games // 2, dtype=jnp.int32),
        ]
    )
    game_index = jnp.arange(num_games)
    search_a = _make_mcts_search(
        graphdef_a,
        game=game,
        num_simulations=replay_simulations,
        gumbel_scale=gumbel_scale,
    )
    search_b = _make_mcts_search(
        graphdef_b,
        game=game,
        num_simulations=replay_simulations,
        gumbel_scale=gumbel_scale,
    )

    @jax.jit
    def replay(
        params_a: nnx.State,
        params_b: nnx.State,
        rng_key: jax.Array,
    ):
        def step(carry, key):
            state, return_a = carry
            active = ~(state.terminated | state.truncated)
            key_a, key_b = jax.random.split(key)
            search_state = _replace_inactive_lanes(state, active, dummy_state)
            action_a = search_a(params_a, search_state, key_a)
            action_b = search_b(params_b, search_state, key_b)
            action = jnp.where(
                state.current_player == player_a_seat,
                action_a,
                action_b,
            )
            stepped_state = jax.vmap(env.step)(state, action)
            state = _replace_inactive_lanes(stepped_state, active, state)
            step_return = stepped_state.rewards[game_index, player_a_seat]
            return_a = return_a + jnp.where(active, step_return, 0.0)
            return (state, return_a), None

        rng_key, init_key, scan_key = jax.random.split(rng_key, 3)
        state = jax.vmap(env.init)(jax.random.split(init_key, num_games))
        dummy_state = jax.vmap(env.init)(
            jax.random.split(jax.random.PRNGKey(0), num_games)
        )
        scan_keys = jax.random.split(scan_key, max_steps)
        (state, return_a), _ = jax.lax.scan(
            step,
            (state, jnp.zeros(num_games)),
            xs=scan_keys[:target_ply],
        )
        return state, return_a, scan_keys[target_ply], player_a_seat

    return replay


def _make_mcts_match(
    graphdef_a: nnx.GraphDef[Net],
    graphdef_b: nnx.GraphDef[Net],
    *,
    game: str,
    num_games: int,
    max_steps: int,
    num_simulations: int,
    gumbel_scale: float,
):
    env = pgx.make(resolve_game(game).env_id)
    player_a_seat = jnp.concatenate(
        [
            jnp.zeros(num_games // 2, dtype=jnp.int32),
            jnp.ones(num_games // 2, dtype=jnp.int32),
        ]
    )
    game_index = jnp.arange(num_games)
    search_a = _make_mcts_search(
        graphdef_a,
        game=game,
        num_simulations=num_simulations,
        gumbel_scale=gumbel_scale,
    )
    search_b = _make_mcts_search(
        graphdef_b,
        game=game,
        num_simulations=num_simulations,
        gumbel_scale=gumbel_scale,
    )

    @jax.jit
    def play(params_a: nnx.State, params_b: nnx.State, rng_key: jax.Array) -> jax.Array:
        def step(carry, key):
            state, return_a = carry
            active = ~(state.terminated | state.truncated)
            key_a, key_b = jax.random.split(key)
            search_state = _replace_inactive_lanes(state, active, dummy_state)
            action_a = search_a(params_a, search_state, key_a)
            action_b = search_b(params_b, search_state, key_b)
            action = jnp.where(
                state.current_player == player_a_seat,
                action_a,
                action_b,
            )
            stepped_state = jax.vmap(env.step)(state, action)
            state = _replace_inactive_lanes(stepped_state, active, state)
            step_return = stepped_state.rewards[game_index, player_a_seat]
            return_a = return_a + jnp.where(active, step_return, 0.0)
            return (state, return_a), None

        rng_key, init_key, scan_key = jax.random.split(rng_key, 3)
        state = jax.vmap(env.init)(jax.random.split(init_key, num_games))
        dummy_state = jax.vmap(env.init)(
            jax.random.split(jax.random.PRNGKey(0), num_games)
        )
        (_, return_a), _ = jax.lax.scan(
            step,
            (state, jnp.zeros(num_games)),
            xs=jax.random.split(scan_key, max_steps),
        )
        wins_a = jnp.sum(return_a == 1.0).astype(jnp.int32)
        draws = jnp.sum(return_a == 0.0).astype(jnp.int32)
        wins_b = jnp.sum(return_a == -1.0).astype(jnp.int32)
        return jnp.stack([wins_a, draws, wins_b])

    return play


def _make_mcts_search_details(
    graphdef: nnx.GraphDef[Net],
    *,
    game: str,
    num_simulations: int,
    gumbel_scale: float,
):
    env = pgx.make(resolve_game(game).env_id)

    def recurrent_fn(params, rng_key, action, state):
        del rng_key
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = apply_model(graphdef, params, state.observation)
        logits = _mask_invalid_logits(logits, state.legal_action_mask)
        done = state.terminated | state.truncated
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(done, 0.0, value)
        discount = jnp.where(done, 0.0, -jnp.ones_like(value))
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, state

    @jax.jit
    def search(params: nnx.State, state, rng_key: jax.Array):
        logits, value = apply_model(graphdef, params, state.observation)
        logits = _mask_invalid_logits(logits, state.legal_action_mask)
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=gumbel_scale,
        )
        return policy_output.action, policy_output.action_weights, value

    return search


def _make_mcts_search(
    graphdef: nnx.GraphDef[Net],
    *,
    game: str,
    num_simulations: int,
    gumbel_scale: float,
):
    env = pgx.make(resolve_game(game).env_id)

    def recurrent_fn(params, rng_key, action, state):
        del rng_key
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = apply_model(graphdef, params, state.observation)
        logits = _mask_invalid_logits(logits, state.legal_action_mask)
        done = state.terminated | state.truncated
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(done, 0.0, value)
        discount = jnp.where(done, 0.0, -jnp.ones_like(value))
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, state

    @jax.jit
    def search(params: nnx.State, state, rng_key: jax.Array) -> jax.Array:
        logits, value = apply_model(graphdef, params, state.observation)
        logits = _mask_invalid_logits(logits, state.legal_action_mask)
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=gumbel_scale,
        )
        return policy_output.action

    return search


def _replace_inactive_lanes(state, active: jax.Array, replacement):
    """Select ``state`` for active batch lanes and ``replacement`` otherwise."""

    def select(state_leaf, replacement_leaf):
        condition = active
        while condition.ndim < state_leaf.ndim:
            condition = condition[..., None]
        return jnp.where(condition, state_leaf, replacement_leaf)

    return jax.tree.map(select, state, replacement)


def _mask_invalid_logits(logits: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def _load_checkpoint(path: Path, *, name: str, game: str) -> _LoadedCheckpoint:
    model = load_checkpoint(path)
    expected_obs_shape = initial_observation_shape(game)
    expected_action_size = make_env(game).num_actions
    if (
        model.config.obs_shape != expected_obs_shape
        or model.config.action_size != expected_action_size
    ):
        msg = (
            f"checkpoint {path} does not match game {resolve_game(game).name!r}: "
            f"obs_shape={model.config.obs_shape}, action_size={model.config.action_size}; "
            f"expected obs_shape={expected_obs_shape}, action_size={expected_action_size}"
        )
        raise ValueError(msg)
    graphdef, params = nnx.split(model, nnx.Param)
    return _LoadedCheckpoint(name=name, path=path, graphdef=graphdef, params=params)


def _checkpoint_names(paths: Sequence[Path]) -> list[str]:
    stems = [path.stem for path in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{index:04d}_{path.stem}" for index, path in enumerate(paths)]


def _pairing_names(
    ordered_names: Sequence[str],
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
    for checkpoint_name in ordered_names:
        if checkpoint_name != anchor_name:
            pairs.add(_ordered_pair(anchor_name, checkpoint_name, order))
    for left, right in zip(ordered_names, ordered_names[1:], strict=False):
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


def _checkpoint_ladder_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"iter_(\d+)\.msgpack", path.name)
    if match is not None:
        return (0, int(match.group(1)), path.name)
    if path.name == "final.msgpack":
        return (1, 0, path.name)
    return (2, 0, path.name)


def _validate_unique_names(names: Sequence[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("names must be unique")


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


def _parse_probe_simulations(raw: str, *, fallback: int) -> list[int]:
    if not raw:
        return [fallback]
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--probe-budgets must be comma-separated integers") from exc
    return _validate_probe_simulations(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a greedy Elo ladder across jaxzero checkpoints."
    )
    parser.add_argument("checkpoints", type=Path, nargs="*")
    parser.add_argument("--game", choices=supported_games(), default=DEFAULT_GAME)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*.msgpack")
    parser.add_argument(
        "--mode",
        choices=("anchored-sequential", "round-robin"),
        default="anchored-sequential",
    )
    parser.add_argument(
        "--games-per-pairing", type=int, default=DEFAULT_GAMES_PER_PAIRING
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--evaluator-mode",
        choices=("greedy", "mcts"),
        default=DEFAULT_EVALUATOR_MODE,
    )
    parser.add_argument(
        "--mcts-simulations", type=int, default=DEFAULT_MCTS_SIMULATIONS
    )
    parser.add_argument("--gumbel-scale", type=float, default=DEFAULT_GUMBEL_SCALE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-iterations", type=int, default=DEFAULT_FIT_ITERATIONS)
    parser.add_argument("--elo-k", type=float, default=DEFAULT_ELO_K)
    parser.add_argument(
        "--trace-plies",
        type=int,
        default=0,
        help="When >0, trace this many opening plies for exactly two checkpoints "
        "instead of fitting an Elo ladder.",
    )
    parser.add_argument(
        "--trace-summary-only",
        action="store_true",
        help="With --trace-plies, omit per-lane trace records and print only summaries.",
    )
    parser.add_argument(
        "--probe-ply",
        type=int,
        default=None,
        help="Replay a two-checkpoint MCTS match to this ply and probe root policy "
        "choices instead of fitting an Elo ladder.",
    )
    parser.add_argument(
        "--probe-budgets",
        default="",
        help="Comma-separated MCTS simulation budgets for --probe-ply. Defaults to "
        "--mcts-simulations.",
    )
    parser.add_argument(
        "--probe-top-k",
        type=int,
        default=5,
        help="Number of top action weights to emit per lane and probe budget.",
    )
    args = parser.parse_args(argv)

    if args.games_per_pairing <= 0:
        parser.error("--games-per-pairing must be positive")
    if args.games_per_pairing % 2 != 0:
        parser.error("--games-per-pairing must be even to balance seats")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.mcts_simulations <= 0:
        parser.error("--mcts-simulations must be positive")
    if args.gumbel_scale < 0.0:
        parser.error("--gumbel-scale must be non-negative")
    if args.fit_iterations <= 0:
        parser.error("--fit-iterations must be positive")
    if args.elo_k <= 0:
        parser.error("--elo-k must be positive")
    if args.trace_plies < 0:
        parser.error("--trace-plies must be non-negative")
    if args.probe_ply is not None and args.probe_ply < 0:
        parser.error("--probe-ply must be non-negative")
    if args.probe_top_k <= 0:
        parser.error("--probe-top-k must be positive")
    try:
        probe_budgets = _parse_probe_simulations(
            args.probe_budgets,
            fallback=args.mcts_simulations,
        )
    except ValueError as exc:
        parser.error(str(exc))

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
    if args.trace_plies > 0 and args.probe_ply is not None:
        parser.error("--trace-plies and --probe-ply are mutually exclusive")
    if args.trace_plies > 0:
        if len(checkpoint_paths) != 2:
            parser.error("--trace-plies requires exactly two checkpoints")
        trace = trace_checkpoint_game(
            checkpoint_paths,
            game=args.game,
            games=args.games_per_pairing,
            max_steps=args.max_steps,
            evaluator_mode=args.evaluator_mode,
            mcts_simulations=args.mcts_simulations,
            gumbel_scale=args.gumbel_scale,
            seed=args.seed,
            trace_plies=args.trace_plies,
            summary_only=args.trace_summary_only,
        )
        print(json.dumps(trace, indent=2, sort_keys=True))
        return 0
    if args.probe_ply is not None:
        if len(checkpoint_paths) != 2:
            parser.error("--probe-ply requires exactly two checkpoints")
        probe = probe_checkpoint_state(
            checkpoint_paths,
            game=args.game,
            games=args.games_per_pairing,
            max_steps=args.max_steps,
            replay_simulations=args.mcts_simulations,
            probe_simulations=probe_budgets,
            gumbel_scale=args.gumbel_scale,
            seed=args.seed,
            target_ply=args.probe_ply,
            top_k=args.probe_top_k,
        )
        print(json.dumps(probe, indent=2, sort_keys=True))
        return 0

    result = evaluate_checkpoint_ladder(
        checkpoint_paths,
        game=args.game,
        games_per_pairing=args.games_per_pairing,
        max_steps=args.max_steps,
        mode=args.mode,
        evaluator_mode=args.evaluator_mode,
        mcts_simulations=args.mcts_simulations,
        gumbel_scale=args.gumbel_scale,
        seed=args.seed,
        fit_iterations=args.fit_iterations,
        elo_k=args.elo_k,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
