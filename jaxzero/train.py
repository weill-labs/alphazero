"""Buffer-free optax training loop and checkpoints for JAX AlphaZero."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import msgpack
import optax
from flax import nnx, serialization

from jaxzero.arena import gating_summary, make_gating_match, update_elo
from jaxzero.evaluate import make_evaluator, vs_random_metrics
from jaxzero.net import (
    ARCH_RESNET,
    ARCH_TRANSFORMER,
    INPUT_EMBED_LINEAR,
    POLICY_HEAD_FLATTEN,
    AlphaZeroNetConfig,
    Net,
    apply_model,
    create_model,
)
from jaxzero.selfplay import (
    SelfPlayConfig,
    SelfPlayData,
    flatten_selfplay_data,
    initial_observation_shape,
    make_env,
    make_selfplay,
    mirror_selfplay_data,
)

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 10
    batch_size: int = 32
    num_simulations: int = 32
    max_steps: int = 64
    channels: int = 64
    num_res_blocks: int = 5
    learning_rate: float = 1e-3
    minibatch_size: int = 1024
    seed: int = 0
    checkpoint_path: str | None = None
    checkpoint_every: int | None = None
    init_checkpoint: str | None = None
    eval_interval: int | None = None
    eval_games: int = 64
    replay_capacity: int | None = None
    gating_interval: int | None = None
    gating_games: int = 20
    gating_threshold: float = 0.55
    value_loss_weight: float = 1.0
    mirror_augment: bool = False
    weight_decay: float = 0.0
    arch: str = ARCH_RESNET
    d_model: int = 128
    num_layers: int = 6
    num_heads: int = 4
    mlp_dim: int = 512
    use_value_cls_token: bool = False
    policy_head_style: str = POLICY_HEAD_FLATTEN
    input_embed_style: str = INPUT_EMBED_LINEAR

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            msg = "iterations must be positive"
            raise ValueError(msg)
        if self.learning_rate <= 0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if self.minibatch_size <= 0:
            msg = "minibatch_size must be positive"
            raise ValueError(msg)
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            msg = "checkpoint_every must be positive when set"
            raise ValueError(msg)
        if self.eval_interval is not None and self.eval_interval <= 0:
            msg = "eval_interval must be positive when set"
            raise ValueError(msg)
        if self.eval_games <= 0:
            msg = "eval_games must be positive"
            raise ValueError(msg)
        if self.replay_capacity is not None and self.replay_capacity <= 0:
            msg = "replay_capacity must be positive when set"
            raise ValueError(msg)
        if self.gating_interval is not None:
            if self.gating_interval <= 0:
                msg = "gating_interval must be positive when set"
                raise ValueError(msg)
            if self.gating_games <= 0:
                msg = "gating_games must be positive"
                raise ValueError(msg)
            if self.gating_games % 2 != 0:
                msg = "gating_games must be even (to balance seatings)"
                raise ValueError(msg)
            if not 0.0 <= self.gating_threshold <= 1.0:
                msg = "gating_threshold must be in [0, 1]"
                raise ValueError(msg)
        if self.value_loss_weight <= 0:
            msg = "value_loss_weight must be positive"
            raise ValueError(msg)
        if self.weight_decay < 0:
            msg = "weight_decay must be non-negative"
            raise ValueError(msg)
        SelfPlayConfig(
            batch_size=self.batch_size,
            num_simulations=self.num_simulations,
            max_steps=self.max_steps,
        )
        AlphaZeroNetConfig(
            obs_shape=initial_observation_shape(),
            action_size=make_env().num_actions,
            channels=self.channels,
            num_res_blocks=self.num_res_blocks,
            arch=self.arch,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            use_value_cls_token=self.use_value_cls_token,
            policy_head_style=self.policy_head_style,
            input_embed_style=self.input_embed_style,
        )


@dataclass(frozen=True)
class TrainingResult:
    config: TrainingConfig
    net_config: AlphaZeroNetConfig
    params: nnx.State
    metrics: list[dict[str, float | int]]
    checkpoint_path: str | None


def build_net_config(config: TrainingConfig) -> AlphaZeroNetConfig:
    env = make_env()
    return AlphaZeroNetConfig(
        obs_shape=initial_observation_shape(),
        action_size=env.num_actions,
        channels=config.channels,
        num_res_blocks=config.num_res_blocks,
        arch=config.arch,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        use_value_cls_token=config.use_value_cls_token,
        policy_head_style=config.policy_head_style,
        input_embed_style=config.input_embed_style,
    )


def _loss(
    graphdef: nnx.GraphDef[Net],
    params: nnx.State,
    batch: SelfPlayData,
    *,
    value_loss_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    policy_logits, value = apply_model(graphdef, params, batch.observation)
    policy_loss_per_example = optax.softmax_cross_entropy(
        policy_logits,
        batch.action_weights,
    )
    policy_loss = jnp.mean(policy_loss_per_example)

    value_mask = batch.value_mask.astype(jnp.float32)
    value_denom = jnp.maximum(jnp.sum(value_mask), 1.0)
    value_loss = (
        jnp.sum(jnp.square(value - batch.value_target) * value_mask) / value_denom
    )

    # `loss` (the gradient signal) is weighted; `policy_loss` and `value_loss`
    # are reported unweighted so wandb curves remain comparable across runs
    # with different weights.
    loss = policy_loss + value_loss_weight * value_loss
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "value_mask_fraction": jnp.mean(value_mask),
    }
    return loss, metrics


def make_update_step(
    graphdef: nnx.GraphDef[Net],
    tx: optax.GradientTransformation,
    *,
    value_loss_weight: float = 1.0,
):
    def loss_fn(params: nnx.State, batch: SelfPlayData):
        return _loss(graphdef, params, batch, value_loss_weight=value_loss_weight)

    @jax.jit
    def update_step(
        params: nnx.State,
        opt_state: optax.OptState,
        batch: SelfPlayData,
    ):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params,
            batch,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {**metrics, "loss": loss}
        return params, opt_state, metrics

    return update_step


def _host_metrics(
    metrics: dict[str, jax.Array], *, iteration: int
) -> dict[str, float | int]:
    return {
        "iteration": iteration,
        **{key: float(jax.device_get(value)) for key, value in metrics.items()},
    }


def _train_epoch(
    update_step,
    params: nnx.State,
    opt_state: optax.OptState,
    data: SelfPlayData,
    minibatch_size: int,
    key: jax.Array,
) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
    """One pass over the iteration's self-play data in fixed-size minibatches.

    A single full-batch gradient over ``batch_size * max_steps`` examples OOMs at
    GPU scale (the backprop activations don't fit). Minibatching bounds each
    step's activation memory and yields several gradient steps per iteration.
    The trailing partial minibatch is dropped so every step shares one shape.
    """
    n = int(data.observation.shape[0])
    size = min(minibatch_size, n)
    num_minibatches = n // size
    perm = jax.random.permutation(key, n)[: num_minibatches * size]
    perm = perm.reshape(num_minibatches, size)

    metric_totals: dict[str, jax.Array] | None = None
    for i in range(num_minibatches):
        minibatch = jax.tree.map(lambda leaf, idx=perm[i]: leaf[idx], data)
        params, opt_state, metrics = update_step(params, opt_state, minibatch)
        if metric_totals is None:
            metric_totals = dict(metrics)
        else:
            metric_totals = {k: metric_totals[k] + v for k, v in metrics.items()}

    assert metric_totals is not None  # num_minibatches >= 1 since size <= n
    mean_metrics = {k: v / num_minibatches for k, v in metric_totals.items()}
    return params, opt_state, mean_metrics


def _append_to_buffer(
    buffer: SelfPlayData | None, new_data: SelfPlayData, capacity: int | None
) -> SelfPlayData:
    """Return the examples to train on this iteration.

    Buffer-free (``capacity`` is None): just ``new_data``. Otherwise keep the
    most recent ``capacity`` examples across iterations and train on those — the
    reused data gives the value head far more signal to calibrate on than a
    single iteration's fresh self-play.
    """
    if capacity is None:
        return new_data
    combined = (
        new_data
        if buffer is None
        else jax.tree.map(
            lambda b, n: jnp.concatenate([b, n], axis=0), buffer, new_data
        )
    )
    n = int(combined.observation.shape[0])
    if n > capacity:
        combined = jax.tree.map(lambda x: x[n - capacity :], combined)
    return combined


def run_training(
    config: TrainingConfig,
    *,
    on_iteration: Callable[[dict[str, float | int]], None] | None = None,
    on_checkpoint: Callable[[str], None] | None = None,
    extra_evaluator: Callable[[Net], dict[str, float]] | None = None,
) -> TrainingResult:
    """Run buffer-free self-play/training for ``config.iterations``.

    Each iteration generates fresh self-play data and takes one pass of
    minibatched gradient steps over it. ``on_iteration`` (if given) receives the
    per-iteration host metrics as they are produced, for live logging. With
    ``config.checkpoint_every`` set, a periodic ``iter_NNNN.msgpack`` is written
    next to ``checkpoint_path`` every N iterations and ``on_checkpoint`` (if
    given) is called with its path — letting a long run be certified mid-flight.
    """

    if config.init_checkpoint is not None:
        # Warm start: continue from a trained net (its stored config wins over
        # config.channels/num_res_blocks) so e.g. a low-sims-bootstrapped net can
        # be refined at high sims without the cold-start uniform-target problem.
        model = load_checkpoint(config.init_checkpoint)
        net_config = model.config
    else:
        net_config = build_net_config(config)
        model = create_model(net_config, seed=config.seed)
    graphdef, params = nnx.split(model, nnx.Param)

    selfplay = make_selfplay(
        SelfPlayConfig(
            batch_size=config.batch_size,
            num_simulations=config.num_simulations,
            max_steps=config.max_steps,
        ),
        graphdef,
    )
    # AdamW when weight_decay > 0 (decoupled L2 reg); standard Adam otherwise.
    # The closed alphago-{ul3, 1q2, 1kc} trail never tried regularization;
    # if the plateau is overfitting (mid-training peak then regression), AdamW
    # at small weight_decay can pull params back toward a simpler hypothesis.
    tx = (
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        if config.weight_decay > 0
        else optax.adam(config.learning_rate)
    )
    opt_state = tx.init(params)
    update_step = make_update_step(
        graphdef, tx, value_loss_weight=config.value_loss_weight
    )
    evaluator = (
        make_evaluator(
            graphdef, num_games=config.eval_games, max_steps=config.max_steps
        )
        if config.eval_interval is not None
        else None
    )
    gating_enabled = config.gating_interval is not None
    gating_match = (
        make_gating_match(
            graphdef,
            num_games=config.gating_games,
            max_steps=config.max_steps,
        )
        if gating_enabled
        else None
    )
    # When gating is on, ``best_params`` is the source for self-play data; when
    # off, self-play uses the live ``params`` (legacy behavior). The closed
    # alphago-{ul3,1q2,1kc} bead trail showed value-MAE plateauing across every
    # config — pinning the data source to a vetted best net is the active lever.
    best_params = jax.tree.map(jnp.copy, params) if gating_enabled else None
    best_elo = 0.0
    last_gating_winrate = 0.0
    last_promoted = 0

    key = jax.random.PRNGKey(config.seed)
    history: list[dict[str, float | int]] = []
    buffer: SelfPlayData | None = None
    for iteration in range(config.iterations):
        key, selfplay_key, shuffle_key = jax.random.split(key, 3)
        selfplay_params = best_params if gating_enabled else params
        new_data = flatten_selfplay_data(selfplay(selfplay_params, selfplay_key))
        if config.mirror_augment:
            new_data = mirror_selfplay_data(new_data)
        buffer = _append_to_buffer(buffer, new_data, config.replay_capacity)
        params, opt_state, metrics = _train_epoch(
            update_step, params, opt_state, buffer, config.minibatch_size, shuffle_key
        )
        host_metrics = _host_metrics(metrics, iteration=iteration)
        if gating_enabled:
            iteration_number = iteration + 1
            if iteration_number % config.gating_interval == 0:
                key, gating_key = jax.random.split(key)
                counts = gating_match(params, best_params, gating_key)
                result = gating_summary(
                    counts,
                    num_games=config.gating_games,
                    threshold=config.gating_threshold,
                )
                last_gating_winrate = result.winrate
                last_promoted = result.promoted
                host_metrics["eval/gating_wins"] = result.wins
                host_metrics["eval/gating_draws"] = result.draws
                host_metrics["eval/gating_losses"] = result.losses
                host_metrics["eval/gating_score"] = result.score
                if result.promoted:
                    best_elo = update_elo(best_elo, best_elo, result.score)
                    best_params = jax.tree.map(jnp.copy, params)
            host_metrics["eval/gating_winrate"] = last_gating_winrate
            host_metrics["eval/promoted"] = last_promoted
            host_metrics["eval/elo"] = best_elo
        if (
            config.eval_interval is not None
            and (iteration + 1) % config.eval_interval == 0
        ):
            # Score the SAME params that get persisted: under gating the
            # checkpoint saves best_params (see the save sites below), so the
            # inline eval must score best_params too, otherwise eval/c4_* and
            # eval/vs_random_* describe the live candidate while the
            # saved/certified checkpoint is a different (best) model. With
            # gating off, best_params is None and we score the live params.
            eval_params = best_params if gating_enabled else params
            if evaluator is not None:
                key, eval_key = jax.random.split(key)
                host_metrics.update(vs_random_metrics(evaluator(eval_params, eval_key)))
            if extra_evaluator is not None:
                host_metrics.update(extra_evaluator(nnx.merge(graphdef, eval_params)))
        history.append(host_metrics)
        if on_iteration is not None:
            on_iteration(host_metrics)

        if (
            config.checkpoint_every is not None
            and config.checkpoint_path is not None
            and (iteration + 1) % config.checkpoint_every == 0
        ):
            periodic_path = str(
                Path(config.checkpoint_path).parent
                / f"iter_{iteration + 1:04d}.msgpack"
            )
            # When gating is on, persist the best (verified-strongest) net,
            # not the live candidate. Otherwise solver-anchored certs of these
            # checkpoints read a net that gating itself flagged as not-better.
            persist_params = best_params if gating_enabled else params
            save_checkpoint(nnx.merge(graphdef, persist_params), periodic_path)
            if on_checkpoint is not None:
                on_checkpoint(periodic_path)

    if config.checkpoint_path is not None:
        # Same rationale as periodic save: persist best when gating is on.
        persist_params = best_params if gating_enabled else params
        save_checkpoint(nnx.merge(graphdef, persist_params), config.checkpoint_path)

    return TrainingResult(
        config=config,
        net_config=net_config,
        params=params,
        metrics=history,
        checkpoint_path=config.checkpoint_path,
    )


def save_checkpoint(model: Net, path: str | Path) -> None:
    """Save a single-file checkpoint containing config and NNX params."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "config": model.config.to_dict(),
        "params": serialization.to_bytes(nnx.to_pure_dict(nnx.state(model, nnx.Param))),
    }
    checkpoint_path.write_bytes(msgpack.packb(payload, use_bin_type=True))


def load_checkpoint(path: str | Path) -> Net:
    """Load a model from a checkpoint created by :func:`save_checkpoint`."""

    payload = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    if not isinstance(payload, dict):
        msg = "checkpoint payload must be a mapping"
        raise ValueError(msg)
    if int(payload["version"]) != CHECKPOINT_VERSION:
        msg = f"unsupported checkpoint version {payload['version']!r}"
        raise ValueError(msg)
    config = AlphaZeroNetConfig.from_dict(payload["config"])
    model = create_model(config, seed=0)
    params = serialization.from_bytes(
        nnx.to_pure_dict(nnx.state(model, nnx.Param)),
        payload["params"],
    )
    nnx.update(model, params)
    return model


def main(argv: list[str] | None = None) -> None:
    from jaxzero.cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()
