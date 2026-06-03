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
    default_arch: str
    default_use_value_cls_token: bool
    default_policy_head_style: str
    default_input_embed_style: str
    supports_solver_eval: bool
    supports_solver_rehearsal: bool
    supports_mirror_augment: bool
    supports_per_column_policy: bool


_SPECS = {
    CONNECT_FOUR: GameSpec(
        name=CONNECT_FOUR,
        env_id="connect_four",
        default_max_steps=64,
        default_arch="resnet",
        default_use_value_cls_token=False,
        default_policy_head_style="flatten",
        default_input_embed_style="linear",
        supports_solver_eval=True,
        supports_solver_rehearsal=True,
        supports_mirror_augment=True,
        supports_per_column_policy=True,
    ),
    OTHELLO: GameSpec(
        name=OTHELLO,
        env_id="othello",
        default_max_steps=128,
        default_arch="transformer",
        default_use_value_cls_token=True,
        default_policy_head_style="flatten",
        default_input_embed_style="conv3x3",
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


def resolve_network_defaults(
    game: str,
    *,
    arch: str | None = None,
    use_value_cls_token: bool | None = None,
    policy_head_style: str | None = None,
    input_embed_style: str | None = None,
) -> dict[str, str | bool]:
    """Resolve game-aware network defaults for user-facing entrypoints."""

    spec = resolve_game(game)
    resolved_arch = spec.default_arch if arch is None else arch
    if resolved_arch == spec.default_arch:
        default_use_value_cls_token = spec.default_use_value_cls_token
        default_policy_head_style = spec.default_policy_head_style
        default_input_embed_style = spec.default_input_embed_style
    else:
        default_use_value_cls_token = False
        default_policy_head_style = "flatten"
        default_input_embed_style = "linear"

    return {
        "arch": resolved_arch,
        "use_value_cls_token": (
            default_use_value_cls_token
            if use_value_cls_token is None
            else use_value_cls_token
        ),
        "policy_head_style": (
            default_policy_head_style
            if policy_head_style is None
            else policy_head_style
        ),
        "input_embed_style": (
            default_input_embed_style
            if input_embed_style is None
            else input_embed_style
        ),
    }
