"""Connect Four solved-ness certification harness.

The certifier compares an agent against the exact solver on a deterministic
sample of non-terminal positions. Short opening positions are always included
in the sample; with the current bounded solver they may be skipped, and they
will start counting automatically once the solver can handle them within the
configured node budget.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from flax import nnx
from pgx.connect_four import GameState as PgxGameState
from pgx.connect_four import State as PgxConnectFourState

from alphazero.c4_solver import NodeBudgetExceeded, solve
from alphazero.games.connectfour import ConnectFour, ConnectFourState
from jaxzero.net import AlphaZeroNet, apply_model
from jaxzero.train import load_checkpoint as load_jax_checkpoint

C4_BOARD_CELLS = 42
DEFAULT_SAMPLE_SIZE = 32
DEFAULT_SIMS = 200
DEFAULT_SOLVER_MAX_NODES = 250_000
DEFAULT_OPENING_DEPTH = 2
DEFAULT_RANDOM_MIN_PLIES = 18
DEFAULT_RANDOM_MAX_PLIES = 38
_ROWS = 6
_COLS = 7
_PGX_EMPTY = -1
_PGX_FIRST_PLAYER = 0
_PGX_SECOND_PLAYER = 1
_CENTER_FIRST_COLS = (3, 2, 4, 1, 5, 0, 6)


class Agent(Protocol):
    """Framework-agnostic Connect Four agent contract."""

    def move(self, state: ConnectFourState) -> int:
        """Return the selected column for ``state``."""

    def value(self, state: ConnectFourState) -> float:
        """Return a value estimate from the player-to-move perspective."""


SearchFn = Callable[
    [nnx.State, PgxConnectFourState, jax.Array],
    tuple[jax.Array, jax.Array],
]
PredictFn = Callable[[nnx.State, jax.Array], jax.Array]


class JaxMCTSAgent:
    """JAX checkpoint agent that searches pgx Connect Four states with mctx."""

    def __init__(
        self,
        model: AlphaZeroNet,
        *,
        sims: int = DEFAULT_SIMS,
        seed: int = 0,
    ) -> None:
        _validate_positive("sims", sims)
        if model.config.obs_shape != (_ROWS, _COLS, 2):
            raise ValueError(
                "JAX Connect Four checkpoints must use pgx observation shape "
                f"{(_ROWS, _COLS, 2)}, got {model.config.obs_shape}"
            )
        if model.config.action_size != _COLS:
            raise ValueError(
                "JAX Connect Four checkpoints must have action_size "
                f"{_COLS}, got {model.config.action_size}"
            )

        self.model = model
        self.sims = sims
        self.seed = seed
        self._graphdef, self._params = nnx.split(model, nnx.Param)
        self._search = _make_search(self._graphdef, sims)
        self._predict = _make_predict(self._graphdef)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        sims: int = DEFAULT_SIMS,
        seed: int = 0,
    ) -> JaxMCTSAgent:
        """Load a Phase-1 JAX checkpoint and wrap it in the Agent protocol."""

        return cls(load_jax_checkpoint(checkpoint), sims=sims, seed=seed)

    def move(self, state: ConnectFourState) -> int:
        legal_moves = ConnectFour().legal_moves(state)
        if not legal_moves:
            raise ValueError("cannot select a move for a terminal/full state")

        action, _ = self._search(
            self._params,
            _batch_pgx_state(solver_state_to_pgx_state(state)),
            self._rng_key(state),
        )
        return int(jax.device_get(action)[0])

    def value(self, state: ConnectFourState) -> float:
        pgx_state = solver_state_to_pgx_state(state)
        value = self._predict(self._params, pgx_state.observation[None, ...])
        return float(jax.device_get(value)[0])

    def _rng_key(self, state: ConnectFourState) -> jax.Array:
        return jax.random.fold_in(jax.random.PRNGKey(self.seed), _state_token(state))


@dataclass(frozen=True)
class PositionCertification:
    solver_value: int
    optimal_moves: tuple[int, ...]
    agent_move: int
    agent_value: float
    agent_outcome: int
    policy_match: bool
    blunder: bool


@dataclass(frozen=True)
class CertificationReport:
    sampled_positions: int
    evaluated_positions: int
    skipped_positions: int
    policy_matches: int
    blunders: int
    policy_match_percent: float
    blunder_rate: float
    value_mae: float
    solved: bool
    records: tuple[PositionCertification, ...]

    def as_dict(self) -> dict[str, bool | float | int]:
        return {
            "sampled_positions": self.sampled_positions,
            "evaluated_positions": self.evaluated_positions,
            "skipped_positions": self.skipped_positions,
            "policy_matches": self.policy_matches,
            "blunders": self.blunders,
            "policy_match_percent": self.policy_match_percent,
            "blunder_rate": self.blunder_rate,
            "value_mae": self.value_mae,
            "solved": self.solved,
        }


def certify_connect_four(
    agent: Agent,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
    solver_max_nodes: int = DEFAULT_SOLVER_MAX_NODES,
    positions: Sequence[ConnectFourState] | None = None,
    opening_depth: int = DEFAULT_OPENING_DEPTH,
) -> CertificationReport:
    """Compare a Connect Four agent's MCTS move and value to solver labels."""

    c4 = ConnectFour()
    _validate_positive("sample_size", sample_size)
    _validate_positive("solver_max_nodes", solver_max_nodes)
    if opening_depth < 0:
        raise ValueError("opening_depth must be non-negative")

    sample = (
        list(positions)
        if positions is not None
        else sample_positions(
            sample_size=sample_size,
            seed=seed,
            opening_depth=opening_depth,
        )
    )

    records: list[PositionCertification] = []
    skipped_positions = 0

    for state in sample:
        if c4.is_terminal(state) or not c4.legal_moves(state):
            skipped_positions += 1
            continue

        try:
            solver_value, optimal_moves = solve(state, max_nodes=solver_max_nodes)
        except NodeBudgetExceeded:
            skipped_positions += 1
            continue
        if not optimal_moves:
            skipped_positions += 1
            continue

        agent_move = int(agent.move(state))
        if agent_move not in c4.legal_moves(state):
            raise ValueError(f"agent selected illegal action {agent_move}")

        try:
            child_value, _ = solve(
                c4.apply_move(state, agent_move),
                max_nodes=solver_max_nodes,
            )
        except NodeBudgetExceeded:
            skipped_positions += 1
            continue

        agent_value = agent.value(state)
        agent_outcome = -child_value
        policy_match = agent_move in optimal_moves
        records.append(
            PositionCertification(
                solver_value=solver_value,
                optimal_moves=tuple(optimal_moves),
                agent_move=agent_move,
                agent_value=float(agent_value),
                agent_outcome=agent_outcome,
                policy_match=policy_match,
                blunder=agent_outcome < solver_value,
            )
        )

    return _report(
        sampled_positions=len(sample),
        skipped_positions=skipped_positions,
        records=records,
    )


def make_solver_evaluator(
    *,
    sample_size: int = 8,
    sims: int = 64,
    seed: int = 0,
    solver_max_nodes: int = DEFAULT_SOLVER_MAX_NODES,
) -> Callable[[AlphaZeroNet], dict[str, float]]:
    """Return a callback that certifies a JAX model against the solver inline.

    Logs live ``eval/c4_blunder_rate`` (the headline strength signal — what
    fraction of MCTS moves were blunders against the exact solver),
    ``eval/c4_policy_match`` (fraction of MCTS moves that matched a solver-
    optimal choice), and ``eval/c4_value_mae`` (the value-head calibration
    error against solver labels — the *actual* C4 plateau bottleneck per the
    closed alphago-{ul3,1q2,1kc} bead trail). All three share the same fixed
    seed so positions are identical across iterations, making each curve
    comparable across training.
    """

    def run(model: AlphaZeroNet) -> dict[str, float]:
        agent = JaxMCTSAgent(model, sims=sims, seed=seed)
        report = certify_connect_four(
            agent,
            sample_size=sample_size,
            seed=seed,
            solver_max_nodes=solver_max_nodes,
        )
        return {
            "eval/c4_blunder_rate": float(report.blunder_rate),
            "eval/c4_policy_match": float(report.policy_match_percent) / 100.0,
            "eval/c4_value_mae": float(report.value_mae),
        }

    return run


def solver_state_to_pgx_state(state: ConnectFourState) -> PgxConnectFourState:
    """Convert the solver's immutable board state to a pgx Connect Four state."""

    board = np.asarray(state.board, dtype=np.int32)
    if board.shape != (_ROWS, _COLS):
        raise ValueError(f"Connect Four board must have shape {(_ROWS, _COLS)}")
    if not np.isin(board, [-1, 0, 1]).all():
        raise ValueError("Connect Four board contains unknown cell values")
    if state.player not in (1, -1):
        raise ValueError(f"Connect Four player must be +1 or -1, got {state.player}")

    pgx_board = np.full((_ROWS, _COLS), _PGX_EMPTY, dtype=np.int32)
    pgx_board[board == 1] = _PGX_FIRST_PLAYER
    pgx_board[board == -1] = _PGX_SECOND_PLAYER

    game = ConnectFour()
    winner = game.winner(state)
    winner_color = _PGX_EMPTY
    if winner == 1:
        winner_color = _PGX_FIRST_PLAYER
    elif winner == -1:
        winner_color = _PGX_SECOND_PLAYER

    color = _solver_player_to_pgx_color(state.player)
    legal_moves = set(game.legal_moves(state))
    rewards = np.zeros(2, dtype=np.float32)
    if winner == 1:
        rewards = np.array([1.0, -1.0], dtype=np.float32)
    elif winner == -1:
        rewards = np.array([-1.0, 1.0], dtype=np.float32)

    return PgxConnectFourState(
        current_player=jnp.asarray(color, dtype=jnp.int32),
        observation=jnp.asarray(_pgx_observation(pgx_board, color)),
        rewards=jnp.asarray(rewards),
        terminated=jnp.asarray(game.is_terminal(state), dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        legal_action_mask=jnp.asarray(
            [col in legal_moves for col in range(_COLS)], dtype=jnp.bool_
        ),
        _step_count=jnp.asarray(int(np.count_nonzero(board)), dtype=jnp.int32),
        _x=PgxGameState(
            color=jnp.asarray(color, dtype=jnp.int32),
            board=jnp.asarray(pgx_board.reshape(-1), dtype=jnp.int32),
            winner=jnp.asarray(winner_color, dtype=jnp.int32),
        ),
    )


def pgx_state_to_solver_state(state: PgxConnectFourState) -> ConnectFourState:
    """Convert a pgx Connect Four state to the solver's board state."""

    pgx_board = np.asarray(jax.device_get(state._x.board), dtype=np.int32).reshape(
        _ROWS, _COLS
    )
    if not np.isin(
        pgx_board, [_PGX_EMPTY, _PGX_FIRST_PLAYER, _PGX_SECOND_PLAYER]
    ).all():
        raise ValueError("pgx Connect Four board contains unknown cell values")

    board = np.zeros((_ROWS, _COLS), dtype=np.int32)
    board[pgx_board == _PGX_FIRST_PLAYER] = 1
    board[pgx_board == _PGX_SECOND_PLAYER] = -1
    return ConnectFourState(
        board=tuple(tuple(int(cell) for cell in row) for row in board),
        player=_pgx_color_to_solver_player(int(jax.device_get(state._x.color))),
    )


def _make_search(
    graphdef: nnx.GraphDef[AlphaZeroNet],
    sims: int,
) -> SearchFn:
    env = pgx.make("connect_four")

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
    def search(
        params: nnx.State,
        state: PgxConnectFourState,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits, value = apply_model(graphdef, params, state.observation)
        logits = _mask_invalid_logits(logits, state.legal_action_mask)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=sims,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action, policy_output.action_weights

    return search


def _make_predict(graphdef: nnx.GraphDef[AlphaZeroNet]) -> PredictFn:
    @jax.jit
    def predict(params: nnx.State, observation: jax.Array) -> jax.Array:
        _, value = apply_model(graphdef, params, observation)
        return value

    return predict


def _mask_invalid_logits(logits: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def _batch_pgx_state(state: PgxConnectFourState) -> PgxConnectFourState:
    return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=0), state)


def _pgx_observation(pgx_board: np.ndarray, color: int) -> np.ndarray:
    opponent = 1 - color
    return np.stack((pgx_board == color, pgx_board == opponent), axis=-1)


def _solver_player_to_pgx_color(player: int) -> int:
    if player == 1:
        return _PGX_FIRST_PLAYER
    if player == -1:
        return _PGX_SECOND_PLAYER
    raise ValueError(f"Connect Four player must be +1 or -1, got {player}")


def _pgx_color_to_solver_player(color: int) -> int:
    if color == _PGX_FIRST_PLAYER:
        return 1
    if color == _PGX_SECOND_PLAYER:
        return -1
    raise ValueError(f"pgx Connect Four color must be 0 or 1, got {color}")


def _state_token(state: ConnectFourState) -> int:
    token = 0
    for row in state.board:
        for cell in row:
            token = ((token * 3) + (cell + 1)) & 0xFFFFFFFF
    return (token ^ (0 if state.player == 1 else 1)) & 0xFFFFFFFF


def sample_positions(
    *,
    sample_size: int,
    seed: int,
    opening_depth: int = DEFAULT_OPENING_DEPTH,
    random_min_plies: int = DEFAULT_RANDOM_MIN_PLIES,
    random_max_plies: int = DEFAULT_RANDOM_MAX_PLIES,
) -> list[ConnectFourState]:
    """Return a deterministic mix of random self-play and short openings."""

    game = ConnectFour()
    _validate_positive("sample_size", sample_size)
    if opening_depth < 0:
        raise ValueError("opening_depth must be non-negative")
    if random_min_plies < 0:
        raise ValueError("random_min_plies must be non-negative")
    if random_max_plies < random_min_plies:
        raise ValueError("random_max_plies must be >= random_min_plies")

    opening_quota = min(sample_size // 4, sample_size)
    random_quota = sample_size - opening_quota
    rng = np.random.default_rng(seed)
    seen: set[ConnectFourState] = set()
    sample: list[ConnectFourState] = []

    sample.extend(
        _random_self_play_positions(
            game,
            count=random_quota,
            rng=rng,
            random_min_plies=random_min_plies,
            random_max_plies=random_max_plies,
            seen=seen,
        )
    )

    for state in _opening_positions(game, max_depth=opening_depth):
        if len(sample) >= sample_size:
            break
        if state not in seen:
            seen.add(state)
            sample.append(state)

    if len(sample) < sample_size:
        sample.extend(
            _random_self_play_positions(
                game,
                count=sample_size - len(sample),
                rng=rng,
                random_min_plies=0,
                random_max_plies=C4_BOARD_CELLS - 1,
                seen=seen,
            )
        )

    return sample[:sample_size]


def _report(
    *,
    sampled_positions: int,
    skipped_positions: int,
    records: Sequence[PositionCertification],
) -> CertificationReport:
    evaluated_positions = len(records)
    policy_matches = sum(1 for record in records if record.policy_match)
    blunders = sum(1 for record in records if record.blunder)
    value_errors = [
        abs(float(record.agent_value) - float(record.solver_value))
        for record in records
    ]
    value_mae = float(np.mean(value_errors)) if value_errors else 0.0
    policy_match_percent = (
        100.0 * policy_matches / evaluated_positions if evaluated_positions else 0.0
    )
    blunder_rate = blunders / evaluated_positions if evaluated_positions else 0.0
    return CertificationReport(
        sampled_positions=sampled_positions,
        evaluated_positions=evaluated_positions,
        skipped_positions=skipped_positions,
        policy_matches=policy_matches,
        blunders=blunders,
        policy_match_percent=policy_match_percent,
        blunder_rate=blunder_rate,
        value_mae=value_mae,
        solved=evaluated_positions > 0 and blunder_rate == 0.0,
        records=tuple(records),
    )


def _random_self_play_positions(
    game: ConnectFour,
    *,
    count: int,
    rng: np.random.Generator,
    random_min_plies: int,
    random_max_plies: int,
    seen: set[ConnectFourState],
) -> list[ConnectFourState]:
    if count <= 0:
        return []

    positions: list[ConnectFourState] = []
    max_attempts = max(count * 100, 100)

    for _ in range(max_attempts):
        state = game.initial_state()
        target_plies = int(rng.integers(random_min_plies, random_max_plies + 1))
        for _ in range(target_plies):
            if game.is_terminal(state):
                break
            legal_moves = game.legal_moves(state)
            state = game.apply_move(state, int(rng.choice(legal_moves)))

        if game.is_terminal(state) or not game.legal_moves(state) or state in seen:
            continue

        seen.add(state)
        positions.append(state)
        if len(positions) == count:
            break

    return positions


def _opening_positions(game: ConnectFour, *, max_depth: int) -> list[ConnectFourState]:
    positions = [game.initial_state()]
    frontier = [game.initial_state()]
    for _ in range(max_depth):
        next_frontier: list[ConnectFourState] = []
        for state in frontier:
            for action in _ordered_legal_moves(game, state):
                child = game.apply_move(state, action)
                if not game.is_terminal(child):
                    positions.append(child)
                    next_frontier.append(child)
        frontier = next_frontier
    return positions


def _ordered_legal_moves(game: ConnectFour, state: ConnectFourState) -> list[int]:
    legal = set(game.legal_moves(state))
    return [action for action in _CENTER_FIRST_COLS if action in legal]


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Certify a Connect Four checkpoint against exact solver labels."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-max-nodes", type=int, default=DEFAULT_SOLVER_MAX_NODES
    )
    parser.add_argument("--opening-depth", type=int, default=DEFAULT_OPENING_DEPTH)
    args = parser.parse_args(argv)

    agent = JaxMCTSAgent.from_checkpoint(
        args.checkpoint,
        sims=args.sims,
        seed=args.seed,
    )
    report = certify_connect_four(
        agent,
        sample_size=args.sample_size,
        seed=args.seed,
        solver_max_nodes=args.solver_max_nodes,
        opening_depth=args.opening_depth,
    )
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
