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


def evaluate_checkpoint_stability(
    checkpoint_paths: Sequence[str | Path],
    *,
    game: str = DEFAULT_GAME,
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    max_steps: int | None = None,
    mode: PairingMode = "anchored-sequential",
    mcts_simulations_list: Sequence[int],
    seeds: Sequence[int],
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    fit_iterations: int = DEFAULT_FIT_ITERATIONS,
    elo_k: float = DEFAULT_ELO_K,
    instability_threshold: float = 0.25,
) -> dict[str, object]:
    """Run MCTS checkpoint Elo across budgets/seeds and summarize sensitivity."""

    budgets = _validate_positive_ints(
        mcts_simulations_list,
        name="mcts_simulations_list",
    )
    seed_values = _validate_ints(seeds, name="seeds")
    if instability_threshold < 0.0:
        raise ValueError("instability_threshold must be non-negative")

    runs: list[dict[str, object]] = []
    resolved_game = ""
    resolved_max_steps = 0
    for budget in budgets:
        for run_seed in seed_values:
            result = evaluate_checkpoint_ladder(
                checkpoint_paths,
                game=game,
                games_per_pairing=games_per_pairing,
                max_steps=max_steps,
                mode=mode,
                evaluator_mode="mcts",
                mcts_simulations=budget,
                gumbel_scale=gumbel_scale,
                seed=run_seed,
                fit_iterations=fit_iterations,
                elo_k=elo_k,
            )
            if not resolved_game:
                resolved_game = result.game
                resolved_max_steps = result.max_steps
            payload = result.as_dict()
            runs.append(
                {
                    "mcts_simulations": budget,
                    "seed": run_seed,
                    "ratings": payload["ratings"],
                    "pairings": payload["pairings"],
                    "best_name": payload["best_name"],
                    "best_elo": payload["best_elo"],
                }
            )

    if not runs:
        raise ValueError("stability sweep produced no runs")

    rating_summary = _summarize_stability_ratings(runs)
    pairing_summary = _summarize_stability_pairings(
        runs,
        instability_threshold=instability_threshold,
    )
    unstable_pairings = [
        pairing
        for pairing in pairing_summary
        if bool(pairing["unstable_verdict"])
        or float(pairing["score_a_range"]) >= instability_threshold
    ]
    return {
        "game": resolved_game,
        "evaluator_mode": "mcts",
        "mode": mode,
        "games_per_pairing": games_per_pairing,
        "max_steps": resolved_max_steps,
        "mcts_simulations": budgets,
        "seeds": seed_values,
        "gumbel_scale": float(gumbel_scale),
        "instability_threshold": float(instability_threshold),
        "stable": not unstable_pairings,
        "runs": runs,
        "rating_summary": rating_summary,
        "pairing_summary": pairing_summary,
        "unstable_pairings": unstable_pairings,
    }


def evaluate_fixed_position_search(
    checkpoint_paths: Sequence[str | Path],
    *,
    game: str = DEFAULT_GAME,
    max_steps: int | None = None,
    num_positions: int,
    min_ply: int = 4,
    max_ply: int | None = None,
    mcts_simulations_list: Sequence[int],
    seeds: Sequence[int],
    position_seed: int = 0,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    teacher_index: int = 0,
    teacher_simulations: int = 0,
    teacher_seed: int = 0,
) -> dict[str, object]:
    """Compare checkpoint search choices on one fixed random-position batch.

    This is a selection-gate diagnostic, not a solver verdict. It removes the
    largest source of noise in pairwise Elo sweeps by holding evaluated states
    fixed while varying checkpoints, search budgets, and evaluator seeds.
    """

    spec = resolve_game(game)
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")
    max_steps = spec.default_max_steps if max_steps is None else max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if min_ply < 0:
        raise ValueError("min_ply must be non-negative")
    max_ply = max_steps - 1 if max_ply is None else max_ply
    if max_ply < min_ply:
        raise ValueError("max_ply must be greater than or equal to min_ply")
    if max_ply >= max_steps:
        raise ValueError("max_ply must be less than max_steps")
    budgets = _validate_positive_ints(
        mcts_simulations_list,
        name="mcts_simulations_list",
    )
    seed_values = _validate_ints(seeds, name="seeds")
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if teacher_simulations < 0:
        raise ValueError("teacher_simulations must be non-negative")

    paths = [Path(path) for path in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must not be empty")
    if teacher_index < 0 or teacher_index >= len(paths):
        raise ValueError("teacher_index must select a checkpoint")
    names = _checkpoint_names(paths)
    checkpoints = [
        _load_checkpoint(path, name=name, game=spec.name)
        for name, path in zip(names, paths, strict=True)
    ]

    rng = np.random.default_rng(position_seed)
    target_plies = rng.integers(
        min_ply,
        max_ply + 1,
        size=num_positions,
        dtype=np.int32,
    )
    sampler = _make_random_position_sampler(
        game=spec.name,
        num_positions=num_positions,
        max_sample_ply=max_ply,
    )
    sampled_state, reached = sampler(
        jax.random.PRNGKey(position_seed),
        jnp.asarray(target_plies, dtype=jnp.int32),
    )
    sampled_state = jax.device_get(sampled_state)
    reached = np.asarray(jax.device_get(reached), dtype=bool)
    legal_action_mask = np.asarray(sampled_state.legal_action_mask, dtype=bool)
    legal_action_counts = np.sum(legal_action_mask, axis=1).astype(np.int32)

    run_shape = (len(budgets), len(seed_values), num_positions)
    actions = np.empty((len(checkpoints), *run_shape), dtype=np.int32)
    selected_weights = np.empty((len(checkpoints), *run_shape), dtype=np.float32)
    root_values = np.empty((len(checkpoints), *run_shape), dtype=np.float32)

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for budget_index, budget in enumerate(budgets):
            search = _make_mcts_search_details(
                checkpoint.graphdef,
                game=spec.name,
                num_simulations=budget,
                gumbel_scale=gumbel_scale,
            )
            for seed_index, search_seed in enumerate(seed_values):
                action, weights, value = search(
                    checkpoint.params,
                    sampled_state,
                    jax.random.PRNGKey(search_seed),
                )
                action_np = np.asarray(jax.device_get(action), dtype=np.int32)
                weights_np = np.asarray(jax.device_get(weights), dtype=np.float32)
                values_np = np.asarray(jax.device_get(value), dtype=np.float32)
                actions[checkpoint_index, budget_index, seed_index] = action_np
                selected_weights[checkpoint_index, budget_index, seed_index] = (
                    weights_np[np.arange(num_positions), action_np]
                )
                root_values[checkpoint_index, budget_index, seed_index] = values_np

    consensus_actions, consensus_votes = _majority_actions(
        actions.reshape((-1, num_positions)),
        action_size=legal_action_mask.shape[1],
    )
    reference_actions, reference_votes = _majority_actions(
        actions[0].reshape((-1, num_positions)),
        action_size=legal_action_mask.shape[1],
    )
    unique_group_indices = _unique_action_profile_indices(actions)
    deduplicated_consensus_actions, deduplicated_consensus_votes = _majority_actions(
        actions[unique_group_indices].reshape((-1, num_positions)),
        action_size=legal_action_mask.shape[1],
    )
    teacher_actions = None
    teacher_weights = None
    teacher_values = None
    if teacher_simulations > 0:
        teacher_checkpoint = checkpoints[teacher_index]
        teacher_search = _make_mcts_search_details(
            teacher_checkpoint.graphdef,
            game=spec.name,
            num_simulations=teacher_simulations,
            gumbel_scale=gumbel_scale,
        )
        teacher_action, weights, value = teacher_search(
            teacher_checkpoint.params,
            sampled_state,
            jax.random.PRNGKey(teacher_seed),
        )
        teacher_actions = np.asarray(jax.device_get(teacher_action), dtype=np.int32)
        teacher_weights_np = np.asarray(jax.device_get(weights), dtype=np.float32)
        teacher_weights = teacher_weights_np[np.arange(num_positions), teacher_actions]
        teacher_values = np.asarray(jax.device_get(value), dtype=np.float32)

    checkpoint_summary = {}
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        checkpoint_summary[checkpoint.name] = _summarize_fixed_position_checkpoint(
            actions[checkpoint_index],
            selected_weights[checkpoint_index],
            root_values[checkpoint_index],
            consensus_actions=consensus_actions,
            deduplicated_consensus_actions=deduplicated_consensus_actions,
            reference_actions=reference_actions,
            teacher_actions=teacher_actions,
            action_size=legal_action_mask.shape[1],
        )

    runs = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for budget_index, budget in enumerate(budgets):
            for seed_index, search_seed in enumerate(seed_values):
                run_actions = actions[checkpoint_index, budget_index, seed_index]
                runs.append(
                    {
                        "checkpoint": checkpoint.name,
                        "mcts_simulations": int(budget),
                        "seed": int(search_seed),
                        "positions": int(num_positions),
                        "consensus_match": float(
                            np.mean(run_actions == consensus_actions)
                        ),
                        "reference_match": float(
                            np.mean(run_actions == reference_actions)
                        ),
                        "deduplicated_consensus_match": float(
                            np.mean(run_actions == deduplicated_consensus_actions)
                        ),
                        "teacher_match": (
                            None
                            if teacher_actions is None
                            else float(np.mean(run_actions == teacher_actions))
                        ),
                        "mean_selected_weight": float(
                            np.mean(
                                selected_weights[
                                    checkpoint_index,
                                    budget_index,
                                    seed_index,
                                ]
                            )
                        ),
                        "mean_root_value": float(
                            np.mean(
                                root_values[
                                    checkpoint_index,
                                    budget_index,
                                    seed_index,
                                ]
                            )
                        ),
                        "action_counts": _value_counts(run_actions),
                    }
                )

    payload: dict[str, object] = {
        "game": spec.name,
        "evaluator_mode": "fixed-position-mcts",
        "num_positions": int(num_positions),
        "max_steps": int(max_steps),
        "min_ply": int(min_ply),
        "max_ply": int(max_ply),
        "position_seed": int(position_seed),
        "mcts_simulations": budgets,
        "seeds": seed_values,
        "gumbel_scale": float(gumbel_scale),
        "reference_checkpoint": checkpoints[0].name,
        "checkpoints": [
            {
                "name": checkpoint.name,
                "path": str(checkpoint.path),
            }
            for checkpoint in checkpoints
        ],
        "position_summary": {
            "reached_positions": int(np.sum(reached)),
            "target_ply_counts": _value_counts(target_plies),
            "mean_target_ply": float(np.mean(target_plies)),
            "mean_legal_actions": float(np.mean(legal_action_counts)),
            "consensus_action_counts": _value_counts(consensus_actions),
            "mean_consensus_vote_fraction": float(
                np.mean(consensus_votes / actions.reshape((-1, num_positions)).shape[0])
            ),
            "reference_action_counts": _value_counts(reference_actions),
            "mean_reference_vote_fraction": float(
                np.mean(
                    reference_votes / actions[0].reshape((-1, num_positions)).shape[0]
                )
            ),
            "deduplicated_consensus_action_counts": _value_counts(
                deduplicated_consensus_actions
            ),
            "deduplicated_checkpoint_groups": [
                [checkpoints[index].name for index in group]
                for group in _unique_action_profile_groups(actions)
            ],
            "mean_deduplicated_consensus_vote_fraction": float(
                np.mean(
                    deduplicated_consensus_votes
                    / actions[unique_group_indices]
                    .reshape((-1, num_positions))
                    .shape[0]
                )
            ),
        },
        "checkpoint_summary": checkpoint_summary,
        "runs": runs,
    }
    if teacher_actions is not None:
        payload["teacher"] = {
            "checkpoint": checkpoints[teacher_index].name,
            "checkpoint_index": int(teacher_index),
            "mcts_simulations": int(teacher_simulations),
            "seed": int(teacher_seed),
            "action_counts": _value_counts(teacher_actions),
            "mean_selected_weight": float(np.mean(teacher_weights)),
            "mean_root_value": float(np.mean(teacher_values)),
        }
        payload["position_summary"]["teacher_action_counts"] = _value_counts(
            teacher_actions
        )
    return payload


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


def evaluate_forced_actions(
    checkpoint_paths: Sequence[str | Path],
    *,
    game: str = DEFAULT_GAME,
    games: int = DEFAULT_GAMES_PER_PAIRING,
    max_steps: int | None = None,
    replay_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    continuation_simulations: int = DEFAULT_MCTS_SIMULATIONS,
    force_actions: Sequence[int] = (),
    force_actor: str = "",
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    seed: int = 0,
    target_ply: int = 0,
) -> dict[str, object]:
    """Force candidate actions at a match ply and continue to terminal states."""

    spec = resolve_game(game)
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) != 2:
        raise ValueError("forced-action eval requires exactly two checkpoint paths")
    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if replay_simulations <= 0:
        raise ValueError("replay_simulations must be positive")
    if continuation_simulations <= 0:
        raise ValueError("continuation_simulations must be positive")
    actions = _validate_force_actions(force_actions)
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")

    max_steps = spec.default_max_steps if max_steps is None else max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if target_ply < 0 or target_ply >= max_steps:
        raise ValueError("target_ply must be in [0, max_steps)")

    names = _checkpoint_names(paths)
    player_a, player_b = (
        _load_checkpoint(path, name=name, game=spec.name)
        for name, path in zip(names, paths, strict=True)
    )
    return evaluate_forced_action_match(
        player_a,
        player_b,
        game=spec.name,
        games=games,
        max_steps=max_steps,
        replay_simulations=replay_simulations,
        continuation_simulations=continuation_simulations,
        force_actions=actions,
        force_actor=force_actor,
        gumbel_scale=gumbel_scale,
        seed=seed,
        target_ply=target_ply,
    )


def evaluate_forced_action_match(
    player_a: _LoadedCheckpoint,
    player_b: _LoadedCheckpoint,
    *,
    game: str,
    games: int,
    max_steps: int,
    replay_simulations: int,
    continuation_simulations: int,
    force_actions: Sequence[int],
    force_actor: str = "",
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    seed: int = 0,
    target_ply: int = 0,
) -> dict[str, object]:
    """Evaluate terminal outcomes after forcing actions for one actor."""

    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        raise ValueError("games must be even to balance seats")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if replay_simulations <= 0:
        raise ValueError("replay_simulations must be positive")
    if continuation_simulations <= 0:
        raise ValueError("continuation_simulations must be positive")
    actions = _validate_force_actions(force_actions)
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if target_ply < 0 or target_ply >= max_steps:
        raise ValueError("target_ply must be in [0, max_steps)")

    force_player, force_actor_name = _resolve_force_actor(
        force_actor,
        player_a.name,
        player_b.name,
    )
    continue_match = _make_forced_action_continuation(
        player_a.graphdef,
        player_b.graphdef,
        game=game,
        num_games=games,
        max_steps=max_steps,
        target_ply=target_ply,
        replay_simulations=replay_simulations,
        continuation_simulations=continuation_simulations,
        gumbel_scale=gumbel_scale,
    )

    action_results = []
    for action in actions:
        raw = continue_match(
            player_a.params,
            player_b.params,
            jax.random.PRNGKey(seed),
            jnp.asarray(action, dtype=jnp.int32),
            jnp.asarray(force_player, dtype=jnp.int32),
        )
        result = {
            name: np.asarray(jax.device_get(value)) for name, value in raw.items()
        }
        forced_mask = result["forced_mask"].astype(bool)
        target_mask = result["target_mask"].astype(bool)
        return_a = result["return_a"]
        default_action = result["default_action"]
        target_default_counts: dict[str, int] = {}
        for lane in np.flatnonzero(target_mask):
            _increment_count(target_default_counts, int(default_action[lane]))
        forced_lanes = np.flatnonzero(forced_mask)
        action_results.append(
            {
                "action": int(action),
                "target_lanes": int(np.sum(target_mask)),
                "forced_lanes": int(np.sum(forced_mask)),
                "illegal_target_lanes": int(np.sum(target_mask & ~forced_mask)),
                "target_default_action_counts": target_default_counts,
                "forced_result": _outcome_summary(
                    return_a[forced_mask],
                    force_player=force_player,
                ),
                "all_result": _outcome_summary(return_a, force_player=force_player),
                "forced_lane_returns_a": [
                    {"lane": int(lane), "return_a": float(return_a[lane])}
                    for lane in forced_lanes
                ],
            }
        )

    return {
        "game": resolve_game(game).name,
        "games": games,
        "max_steps": max_steps,
        "seed": seed,
        "target_ply": target_ply,
        "replay_simulations": replay_simulations,
        "continuation_simulations": continuation_simulations,
        "gumbel_scale": float(gumbel_scale),
        "player_a": player_a.name,
        "player_b": player_b.name,
        "force_actor": force_actor_name,
        "force_actor_role": "player_a" if force_player == 0 else "player_b",
        "actions": action_results,
    }


def _validate_force_actions(force_actions: Sequence[int]) -> list[int]:
    actions = [int(value) for value in force_actions]
    if not actions:
        raise ValueError("force_actions must not be empty")
    if any(value < 0 for value in actions):
        raise ValueError("force_actions values must be non-negative")
    return actions


def _resolve_force_actor(
    force_actor: str,
    player_a_name: str,
    player_b_name: str,
) -> tuple[int, str]:
    if force_actor in ("", "player_a", player_a_name):
        return 0, player_a_name
    if force_actor in ("player_b", player_b_name):
        return 1, player_b_name
    msg = (
        "force_actor must be empty, 'player_a', 'player_b', "
        f"{player_a_name!r}, or {player_b_name!r}"
    )
    raise ValueError(msg)


def _outcome_summary(
    values: np.ndarray, *, force_player: int
) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "games": 0,
            "wins_a": 0,
            "draws": 0,
            "wins_b": 0,
            "score_a": 0.0,
            "force_actor_score": 0.0,
        }
    wins_a = int(np.sum(values == 1.0))
    draws = int(np.sum(values == 0.0))
    wins_b = int(np.sum(values == -1.0))
    score_a = (wins_a + 0.5 * draws) / int(values.size)
    force_actor_score = score_a if force_player == 0 else 1.0 - score_a
    return {
        "games": int(values.size),
        "wins_a": wins_a,
        "draws": draws,
        "wins_b": wins_b,
        "score_a": float(score_a),
        "force_actor_score": float(force_actor_score),
    }


def _summarize_fixed_position_checkpoint(
    actions: np.ndarray,
    selected_weights: np.ndarray,
    root_values: np.ndarray,
    *,
    consensus_actions: np.ndarray,
    deduplicated_consensus_actions: np.ndarray,
    reference_actions: np.ndarray,
    teacher_actions: np.ndarray | None,
    action_size: int,
) -> dict[str, float | int | dict[str, int] | None]:
    budget_count, seed_count, num_positions = actions.shape
    flat_actions = actions.reshape((budget_count * seed_count, num_positions))
    majority_actions, majority_votes = _majority_actions(
        flat_actions,
        action_size=action_size,
    )
    budget_sensitive = []
    for seed_index in range(seed_count):
        for position_index in range(num_positions):
            budget_actions = {
                int(actions[budget_index, seed_index, position_index])
                for budget_index in range(budget_count)
            }
            budget_sensitive.append(len(budget_actions) > 1)

    seed_sensitive = []
    for budget_index in range(budget_count):
        for position_index in range(num_positions):
            seed_actions = {
                int(actions[budget_index, seed_index, position_index])
                for seed_index in range(seed_count)
            }
            seed_sensitive.append(len(seed_actions) > 1)
    return {
        "action_stability": float(np.mean(majority_votes / flat_actions.shape[0])),
        "stable_position_fraction": float(
            np.mean(majority_votes == flat_actions.shape[0])
        ),
        "budget_sensitive_fraction": float(np.mean(budget_sensitive)),
        "seed_sensitive_fraction": float(np.mean(seed_sensitive)),
        "consensus_match": float(np.mean(flat_actions == consensus_actions)),
        "deduplicated_consensus_match": float(
            np.mean(flat_actions == deduplicated_consensus_actions)
        ),
        "reference_match": float(np.mean(flat_actions == reference_actions)),
        "teacher_match": (
            None
            if teacher_actions is None
            else float(np.mean(flat_actions == teacher_actions))
        ),
        "mean_selected_weight": float(np.mean(selected_weights)),
        "mean_root_value": float(np.mean(root_values)),
        "majority_action_counts": _value_counts(majority_actions),
    }


def _summarize_stability_ratings(
    runs: Sequence[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    names = sorted(
        {
            name
            for run in runs
            for name in (run["ratings"]).keys()  # type: ignore[union-attr]
        }
    )
    summary = {}
    for name in names:
        values = np.asarray(
            [float((run["ratings"])[name]) for run in runs],  # type: ignore[index]
            dtype=np.float64,
        )
        summary[name] = {
            "mean_elo": float(np.mean(values)),
            "std_elo": float(np.std(values)),
            "min_elo": float(np.min(values)),
            "max_elo": float(np.max(values)),
            "elo_range": float(np.max(values) - np.min(values)),
            "best_count": int(sum(run["best_name"] == name for run in runs)),
        }
    return summary


def _summarize_stability_pairings(
    runs: Sequence[dict[str, object]],
    *,
    instability_threshold: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run in runs:
        for pairing in run["pairings"]:  # type: ignore[union-attr]
            key = (str(pairing["player_a"]), str(pairing["player_b"]))
            grouped.setdefault(key, []).append(
                {
                    "mcts_simulations": run["mcts_simulations"],
                    "seed": run["seed"],
                    "score_a": float(pairing["score_a"]),
                    "wins_a": int(pairing["wins_a"]),
                    "draws": int(pairing["draws"]),
                    "wins_b": int(pairing["wins_b"]),
                    "games": int(pairing["games"]),
                }
            )

    summaries = []
    for (player_a, player_b), entries in sorted(grouped.items()):
        scores = np.asarray([entry["score_a"] for entry in entries], dtype=np.float64)
        verdict_counts = {"player_a": 0, "tie": 0, "player_b": 0}
        for score in scores:
            verdict_counts[_score_verdict(float(score))] += 1
        non_tie_verdicts = {
            verdict
            for verdict, count in verdict_counts.items()
            if verdict != "tie" and count
        }
        score_range = float(np.max(scores) - np.min(scores))
        summaries.append(
            {
                "player_a": player_a,
                "player_b": player_b,
                "mean_score_a": float(np.mean(scores)),
                "min_score_a": float(np.min(scores)),
                "max_score_a": float(np.max(scores)),
                "score_a_range": score_range,
                "unstable_verdict": len(non_tie_verdicts) > 1,
                "exceeds_threshold": score_range >= instability_threshold,
                "verdict_counts": verdict_counts,
                "runs": entries,
            }
        )
    return summaries


def _score_verdict(score: float) -> str:
    if score > 0.5:
        return "player_a"
    if score < 0.5:
        return "player_b"
    return "tie"


def _validate_positive_ints(values: Sequence[int], *, name: str) -> list[int]:
    parsed = [int(value) for value in values]
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0 for value in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed


def _validate_ints(values: Sequence[int], *, name: str) -> list[int]:
    parsed = [int(value) for value in values]
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _validate_probe_simulations(probe_simulations: Sequence[int]) -> list[int]:
    return _validate_positive_ints(probe_simulations, name="probe_simulations")


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


def _unique_action_profile_groups(actions: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = []
    profile_to_group: dict[bytes, int] = {}
    for checkpoint_index, checkpoint_actions in enumerate(actions):
        profile = np.ascontiguousarray(checkpoint_actions).tobytes()
        group_index = profile_to_group.get(profile)
        if group_index is None:
            profile_to_group[profile] = len(groups)
            groups.append([checkpoint_index])
        else:
            groups[group_index].append(checkpoint_index)
    return groups


def _unique_action_profile_indices(actions: np.ndarray) -> list[int]:
    return [group[0] for group in _unique_action_profile_groups(actions)]


def _majority_actions(
    actions: np.ndarray,
    *,
    action_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if actions.ndim != 2:
        raise ValueError("actions must have shape [runs, positions]")
    if action_size <= 0:
        raise ValueError("action_size must be positive")
    counts = np.zeros((actions.shape[1], action_size), dtype=np.int32)
    for run_actions in actions:
        np.add.at(counts, (np.arange(actions.shape[1]), run_actions), 1)
    majority_actions = np.argmax(counts, axis=1).astype(np.int32)
    majority_votes = np.max(counts, axis=1).astype(np.int32)
    return majority_actions, majority_votes


def _value_counts(values: np.ndarray | Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in np.asarray(values).reshape(-1):
        _increment_count(counts, int(value))
    return counts


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


def _make_random_position_sampler(
    *,
    game: str,
    num_positions: int,
    max_sample_ply: int,
):
    env = pgx.make(resolve_game(game).env_id)
    neg_inf = jnp.finfo(jnp.float32).min

    @jax.jit
    def sample(rng_key: jax.Array, target_plies: jax.Array):
        def step(carry, inputs):
            state, sampled_state, reached = carry
            ply, key = inputs
            active = ~(state.terminated | state.truncated)
            capture = (target_plies == ply) & active
            sampled_state = _replace_inactive_lanes(state, capture, sampled_state)
            reached = reached | capture

            scores = jax.random.uniform(key, state.legal_action_mask.shape)
            action = jnp.argmax(
                jnp.where(state.legal_action_mask, scores, neg_inf),
                axis=-1,
            )
            stepped_state = jax.vmap(env.step)(state, action)
            state = _replace_inactive_lanes(stepped_state, active, state)
            return (state, sampled_state, reached), None

        rng_key, init_key, scan_key = jax.random.split(rng_key, 3)
        state = jax.vmap(env.init)(jax.random.split(init_key, num_positions))
        sampled_state = state
        reached = target_plies == 0
        scan_inputs = (
            jnp.arange(max_sample_ply + 1, dtype=jnp.int32),
            jax.random.split(scan_key, max_sample_ply + 1),
        )
        (_, sampled_state, reached), _ = jax.lax.scan(
            step,
            (state, sampled_state, reached),
            scan_inputs,
        )
        return sampled_state, reached

    return sample


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


def _make_forced_action_continuation(
    graphdef_a: nnx.GraphDef[Net],
    graphdef_b: nnx.GraphDef[Net],
    *,
    game: str,
    num_games: int,
    max_steps: int,
    target_ply: int,
    replay_simulations: int,
    continuation_simulations: int,
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
    replay_search_a = _make_mcts_search(
        graphdef_a,
        game=game,
        num_simulations=replay_simulations,
        gumbel_scale=gumbel_scale,
    )
    replay_search_b = _make_mcts_search(
        graphdef_b,
        game=game,
        num_simulations=replay_simulations,
        gumbel_scale=gumbel_scale,
    )
    continuation_search_a = _make_mcts_search(
        graphdef_a,
        game=game,
        num_simulations=continuation_simulations,
        gumbel_scale=gumbel_scale,
    )
    continuation_search_b = _make_mcts_search(
        graphdef_b,
        game=game,
        num_simulations=continuation_simulations,
        gumbel_scale=gumbel_scale,
    )

    @jax.jit
    def continue_match(
        params_a: nnx.State,
        params_b: nnx.State,
        rng_key: jax.Array,
        forced_action: jax.Array,
        force_player: jax.Array,
    ) -> dict[str, jax.Array]:
        def replay_step(carry, key):
            state, return_a = carry
            active = ~(state.terminated | state.truncated)
            key_a, key_b = jax.random.split(key)
            search_state = _replace_inactive_lanes(state, active, dummy_state)
            action_a = replay_search_a(params_a, search_state, key_a)
            action_b = replay_search_b(params_b, search_state, key_b)
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

        def continuation_step(carry, key):
            state, return_a = carry
            active = ~(state.terminated | state.truncated)
            key_a, key_b = jax.random.split(key)
            search_state = _replace_inactive_lanes(state, active, dummy_state)
            action_a = continuation_search_a(params_a, search_state, key_a)
            action_b = continuation_search_b(params_b, search_state, key_b)
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
            replay_step,
            (state, jnp.zeros(num_games)),
            xs=scan_keys[:target_ply],
        )

        target_active = ~(state.terminated | state.truncated)
        key_a, key_b = jax.random.split(scan_keys[target_ply])
        search_state = _replace_inactive_lanes(state, target_active, dummy_state)
        action_a = continuation_search_a(params_a, search_state, key_a)
        action_b = continuation_search_b(params_b, search_state, key_b)
        default_action = jnp.where(
            state.current_player == player_a_seat,
            action_a,
            action_b,
        )
        actor_is_a = state.current_player == player_a_seat
        target_actor_mask = jnp.where(force_player == 0, actor_is_a, ~actor_is_a)
        target_mask = target_active & target_actor_mask
        forced_legal = state.legal_action_mask[
            jnp.arange(state.legal_action_mask.shape[0]),
            forced_action,
        ]
        forced_mask = target_mask & forced_legal
        target_action = jnp.where(forced_mask, forced_action, default_action)
        stepped_state = jax.vmap(env.step)(state, target_action)
        state = _replace_inactive_lanes(stepped_state, target_active, state)
        step_return = stepped_state.rewards[game_index, player_a_seat]
        return_a = return_a + jnp.where(target_active, step_return, 0.0)

        (state, return_a), _ = jax.lax.scan(
            continuation_step,
            (state, return_a),
            xs=scan_keys[target_ply + 1 :],
        )
        return {
            "return_a": return_a,
            "target_mask": target_mask,
            "forced_mask": forced_mask,
            "default_action": default_action,
            "target_action": target_action,
        }

    return continue_match


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


def _parse_force_actions(raw: str) -> list[int]:
    if not raw:
        raise ValueError("--force-actions is required with --force-ply")
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--force-actions must be comma-separated integers") from exc
    return _validate_force_actions(values)


def _parse_stability_budgets(raw: str) -> list[int]:
    if not raw:
        return []
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "--stability-budgets must be comma-separated integers"
        ) from exc
    return _validate_positive_ints(values, name="stability_budgets")


def _parse_stability_seeds(raw: str) -> list[int]:
    if not raw:
        raise ValueError("--stability-seeds must not be empty")
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--stability-seeds must be comma-separated integers") from exc
    return _validate_ints(values, name="stability_seeds")


def _parse_position_budgets(raw: str, *, fallback: int) -> list[int]:
    if not raw:
        return [fallback]
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--position-budgets must be comma-separated integers") from exc
    return _validate_positive_ints(values, name="position_budgets")


def _parse_position_seeds(raw: str) -> list[int]:
    if not raw:
        raise ValueError("--position-seeds must not be empty")
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--position-seeds must be comma-separated integers") from exc
    return _validate_ints(values, name="position_seeds")


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
    parser.add_argument(
        "--force-ply",
        type=int,
        default=None,
        help="Replay a two-checkpoint MCTS match to this ply, force each action "
        "from --force-actions on --force-actor lanes, and continue to terminal.",
    )
    parser.add_argument(
        "--force-actions",
        default="",
        help="Comma-separated action ids to evaluate with --force-ply.",
    )
    parser.add_argument(
        "--force-actor",
        default="",
        help="Actor to force: empty/player_a, player_b, or a checkpoint name.",
    )
    parser.add_argument(
        "--continuation-simulations",
        type=int,
        default=DEFAULT_MCTS_SIMULATIONS,
        help="MCTS simulations for the forced ply and remaining continuation.",
    )
    parser.add_argument(
        "--stability-budgets",
        default="",
        help="Comma-separated MCTS simulation budgets for a stability sweep. "
        "When set, runs MCTS Elo once per budget/seed and reports sensitivity.",
    )
    parser.add_argument(
        "--stability-seeds",
        default="0",
        help="Comma-separated evaluator seeds for --stability-budgets.",
    )
    parser.add_argument(
        "--stability-score-threshold",
        type=float,
        default=0.25,
        help="Pair score range that marks a stability-sweep pairing as unstable.",
    )
    parser.add_argument(
        "--position-samples",
        type=int,
        default=0,
        help="When >0, sample this many fixed random positions and compare "
        "checkpoint MCTS action stability/agreement on that shared state batch.",
    )
    parser.add_argument(
        "--position-min-ply",
        type=int,
        default=4,
        help="Earliest random-play ply to sample for --position-samples.",
    )
    parser.add_argument(
        "--position-max-ply",
        type=int,
        default=None,
        help="Latest random-play ply to sample for --position-samples. Defaults "
        "to max_steps - 1.",
    )
    parser.add_argument(
        "--position-budgets",
        default="",
        help="Comma-separated MCTS simulation budgets for --position-samples. "
        "Defaults to --mcts-simulations.",
    )
    parser.add_argument(
        "--position-seeds",
        default="0",
        help="Comma-separated evaluator seeds for --position-samples.",
    )
    parser.add_argument(
        "--position-seed",
        type=int,
        default=0,
        help="Seed for the fixed random-position batch.",
    )
    parser.add_argument(
        "--position-teacher-index",
        type=int,
        default=0,
        help="Checkpoint index to use as the high-budget teacher target for "
        "--position-samples.",
    )
    parser.add_argument(
        "--position-teacher-simulations",
        type=int,
        default=0,
        help="When >0, evaluate the teacher checkpoint with this MCTS budget "
        "on the fixed positions and report teacher-action agreement.",
    )
    parser.add_argument(
        "--position-teacher-seed",
        type=int,
        default=0,
        help="Evaluator seed for --position-teacher-simulations.",
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
    if args.force_ply is not None and args.force_ply < 0:
        parser.error("--force-ply must be non-negative")
    if args.continuation_simulations <= 0:
        parser.error("--continuation-simulations must be positive")
    if args.stability_score_threshold < 0.0:
        parser.error("--stability-score-threshold must be non-negative")
    if args.position_samples < 0:
        parser.error("--position-samples must be non-negative")
    if args.position_min_ply < 0:
        parser.error("--position-min-ply must be non-negative")
    if args.position_max_ply is not None and args.position_max_ply < 0:
        parser.error("--position-max-ply must be non-negative")
    if args.position_teacher_index < 0:
        parser.error("--position-teacher-index must be non-negative")
    if args.position_teacher_simulations < 0:
        parser.error("--position-teacher-simulations must be non-negative")
    try:
        probe_budgets = _parse_probe_simulations(
            args.probe_budgets,
            fallback=args.mcts_simulations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        force_actions = (
            _parse_force_actions(args.force_actions)
            if args.force_ply is not None
            else []
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        stability_budgets = _parse_stability_budgets(args.stability_budgets)
        stability_seeds = _parse_stability_seeds(args.stability_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        position_budgets = _parse_position_budgets(
            args.position_budgets,
            fallback=args.mcts_simulations,
        )
        position_seeds = _parse_position_seeds(args.position_seeds)
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
    active_modes = sum(
        [
            int(args.trace_plies > 0),
            int(args.probe_ply is not None),
            int(args.force_ply is not None),
            int(bool(stability_budgets)),
            int(args.position_samples > 0),
        ]
    )
    if active_modes > 1:
        parser.error(
            "--trace-plies, --probe-ply, --force-ply, and --stability-budgets "
            "are mutually exclusive with --position-samples"
        )
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
    if args.force_ply is not None:
        if len(checkpoint_paths) != 2:
            parser.error("--force-ply requires exactly two checkpoints")
        forced = evaluate_forced_actions(
            checkpoint_paths,
            game=args.game,
            games=args.games_per_pairing,
            max_steps=args.max_steps,
            replay_simulations=args.mcts_simulations,
            continuation_simulations=args.continuation_simulations,
            force_actions=force_actions,
            force_actor=args.force_actor,
            gumbel_scale=args.gumbel_scale,
            seed=args.seed,
            target_ply=args.force_ply,
        )
        print(json.dumps(forced, indent=2, sort_keys=True))
        return 0
    if stability_budgets:
        stability = evaluate_checkpoint_stability(
            checkpoint_paths,
            game=args.game,
            games_per_pairing=args.games_per_pairing,
            max_steps=args.max_steps,
            mode=args.mode,
            mcts_simulations_list=stability_budgets,
            seeds=stability_seeds,
            gumbel_scale=args.gumbel_scale,
            fit_iterations=args.fit_iterations,
            elo_k=args.elo_k,
            instability_threshold=args.stability_score_threshold,
        )
        print(json.dumps(stability, indent=2, sort_keys=True))
        return 0
    if args.position_samples > 0:
        fixed_positions = evaluate_fixed_position_search(
            checkpoint_paths,
            game=args.game,
            max_steps=args.max_steps,
            num_positions=args.position_samples,
            min_ply=args.position_min_ply,
            max_ply=args.position_max_ply,
            mcts_simulations_list=position_budgets,
            seeds=position_seeds,
            position_seed=args.position_seed,
            gumbel_scale=args.gumbel_scale,
            teacher_index=args.position_teacher_index,
            teacher_simulations=args.position_teacher_simulations,
            teacher_seed=args.position_teacher_seed,
        )
        print(json.dumps(fixed_positions, indent=2, sort_keys=True))
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
