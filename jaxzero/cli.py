"""Command-line entrypoint for the JAX AlphaZero trainer."""

from __future__ import annotations

import argparse
import json

from jaxzero.train import TrainingConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train JAX AlphaZero on pgx Connect Four."
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sims", "--num-simulations", dest="num_simulations", type=int, default=32
    )
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", dest="checkpoint_path")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="save iter_NNNN.msgpack next to --checkpoint every N iterations",
    )
    parser.add_argument(
        "--init-checkpoint",
        dest="init_checkpoint",
        help="warm-start training from this checkpoint instead of random init",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        iterations=args.iterations,
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        max_steps=args.max_steps,
        channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        init_checkpoint=args.init_checkpoint,
    )
    result = run_training(config)
    for metrics in result.metrics:
        print(json.dumps(metrics, sort_keys=True))
    if result.checkpoint_path is not None:
        print(json.dumps({"checkpoint": result.checkpoint_path}, sort_keys=True))


if __name__ == "__main__":
    main()
