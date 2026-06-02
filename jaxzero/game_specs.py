"""Game metadata for pgx-backed jaxzero training."""

from __future__ import annotations

from dataclasses import dataclass

CONNECT_FOUR = "connectfour"
OTHELLO = "othello"
DEFAULT_GAME = CONNECT_FOUR


@dataclass(frozen=True)
class GameSpec:
    """Static game capabilities used to wire generic training safely."""

    name: str
    env_id: str
    default_max_steps: int
    supports_solver_eval: bool
    supports_solver_rehearsal: bool
    supports_mirror_augment: bool
    supports_per_column_policy: bool


_SPECS = {
    CONNECT_FOUR: GameSpec(
        name=CONNECT_FOUR,
        env_id="connect_four",
        default_max_steps=64,
        supports_solver_eval=True,
        supports_solver_rehearsal=True,
        supports_mirror_augment=True,
        supports_per_column_policy=True,
    ),
    OTHELLO: GameSpec(
        name=OTHELLO,
        env_id="othello",
        default_max_steps=128,
        supports_solver_eval=False,
        supports_solver_rehearsal=False,
        supports_mirror_augment=False,
        supports_per_column_policy=False,
    ),
}

_ALIASES = {
    CONNECT_FOUR: CONNECT_FOUR,
    "connect_four": CONNECT_FOUR,
    "connect-4": CONNECT_FOUR,
    "connect_4": CONNECT_FOUR,
    "connect4": CONNECT_FOUR,
    "c4": CONNECT_FOUR,
    OTHELLO: OTHELLO,
}


def supported_games() -> tuple[str, ...]:
    """Return canonical game names accepted by user-facing entrypoints."""

    return tuple(_SPECS)


def resolve_game(game: str) -> GameSpec:
    """Return the canonical game spec for ``game`` or raise ``ValueError``."""

    key = game.strip().lower().replace("-", "_")
    try:
        return _SPECS[_ALIASES[key]]
    except KeyError as exc:
        supported = ", ".join(supported_games())
        msg = f"unsupported game {game!r}; supported games: {supported}"
        raise ValueError(msg) from exc
