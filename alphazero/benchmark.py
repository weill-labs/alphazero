"""Benchmark and profile AlphaZero self-play and training throughput."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from alphazero.games import GAME_CHOICES, game_from_name
from alphazero.network import AlphaZeroNet
from alphazero.selfplay import SelfPlayExample, play_game
from alphazero.train import make_optimizer, train_iteration


@dataclass(frozen=True)
class BenchmarkConfig:
    game_name: str = "tictactoe"
    self_play_games: int = 2
    mcts_simulations: int = 32
    mcts_batch_size: int = 16
    train_epochs: int = 4
    batch_size: int = 32
    seed: int = 0
    dirichlet_eps: float = 0.0
    lr: float = 1e-3
    l2_reg: float = 0.0
    device: str = "cpu"
    torch_threads: int | None = None


@dataclass(frozen=True)
class ComponentBreakdown:
    self_play_seconds: float
    network_inference_seconds: float
    mcts_non_inference_seconds: float
    train_step_seconds: float
    train_iteration_seconds: float


@dataclass(frozen=True)
class Throughput:
    self_play_games_per_sec: float
    mcts_net_evals_per_sec: float
    network_inference_evals_per_sec: float
    train_steps_per_sec: float


@dataclass(frozen=True)
class BenchmarkResult:
    config: BenchmarkConfig
    examples: int
    network_evals: int
    train_steps: int
    breakdown: ComponentBreakdown
    throughput: Throughput

    def additive_component_seconds(self) -> dict[str, float]:
        """Return non-overlapping time buckets for dominance comparisons."""

        return {
            "self-play MCTS overhead": self.breakdown.mcts_non_inference_seconds,
            "MCTS network inference": self.breakdown.network_inference_seconds,
            "train step": self.breakdown.train_step_seconds,
        }


class _TimingRecorder:
    def __init__(self) -> None:
        self.seconds: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)

    def record(self, component: str, seconds: float) -> None:
        self.seconds[component] += seconds
        self.counts[component] += 1


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a fixed AlphaZero workload and return timing/throughput metrics."""

    _validate_config(config)
    previous_threads = torch.get_num_threads()
    if config.torch_threads is not None:
        torch.set_num_threads(config.torch_threads)

    try:
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        game = game_from_name(config.game_name)
        device = torch.device(config.device)
        net = AlphaZeroNet(game.num_planes, game.board_shape, game.action_size)
        net.to(device)
        optimizer = make_optimizer(net, optimizer_name="adam", lr=config.lr)
        recorder = _TimingRecorder()

        examples: list[SelfPlayExample] = []
        self_play_started = time.perf_counter()
        for _ in range(config.self_play_games):
            examples.extend(
                play_game(
                    net,
                    game,
                    {
                        "num_simulations": config.mcts_simulations,
                        "batch_size": config.mcts_batch_size,
                        "dirichlet_eps": config.dirichlet_eps,
                        "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
                    },
                    temperature_schedule=1.0,
                    timing_hook=recorder.record,
                )
            )
        self_play_seconds = time.perf_counter() - self_play_started

        train_started = time.perf_counter()
        train_metrics = train_iteration(
            net,
            examples,
            optimizer=optimizer,
            batch_size=config.batch_size,
            epochs=config.train_epochs,
            lr=config.lr,
            l2_reg=config.l2_reg,
            device=device,
            shuffle=True,
            rng=rng,
            timing_hook=recorder.record,
        )
        train_iteration_seconds = time.perf_counter() - train_started

        network_seconds = recorder.seconds["network_inference"]
        train_step_seconds = recorder.seconds["train_step"]
        network_evals = recorder.counts["network_inference"]
        train_steps = int(train_metrics["num_batches"])
        mcts_non_inference_seconds = max(self_play_seconds - network_seconds, 0.0)

        return BenchmarkResult(
            config=config,
            examples=len(examples),
            network_evals=network_evals,
            train_steps=train_steps,
            breakdown=ComponentBreakdown(
                self_play_seconds=self_play_seconds,
                network_inference_seconds=network_seconds,
                mcts_non_inference_seconds=mcts_non_inference_seconds,
                train_step_seconds=train_step_seconds,
                train_iteration_seconds=train_iteration_seconds,
            ),
            throughput=Throughput(
                self_play_games_per_sec=_rate(
                    config.self_play_games, self_play_seconds
                ),
                mcts_net_evals_per_sec=_rate(network_evals, self_play_seconds),
                network_inference_evals_per_sec=_rate(network_evals, network_seconds),
                train_steps_per_sec=_rate(train_steps, train_iteration_seconds),
            ),
        )
    finally:
        if config.torch_threads is not None:
            torch.set_num_threads(previous_threads)


def dominant_cost(result: BenchmarkResult) -> tuple[str, float]:
    components = result.additive_component_seconds()
    return max(components.items(), key=lambda item: item[1])


def format_report(result: BenchmarkResult) -> str:
    dominant_label, dominant_seconds = dominant_cost(result)
    additive_total = sum(result.additive_component_seconds().values())
    dominant_percent = (
        100.0 * dominant_seconds / additive_total if additive_total > 0 else 0.0
    )
    breakdown = result.breakdown
    throughput = result.throughput

    return "\n".join(
        [
            "AlphaZero benchmark",
            f"game: {result.config.game_name}",
            (
                "workload: "
                f"self_play_games={result.config.self_play_games}, "
                f"mcts_sims={result.config.mcts_simulations}, "
                f"mcts_batch_size={result.config.mcts_batch_size}, "
                f"train_epochs={result.config.train_epochs}, "
                f"batch_size={result.config.batch_size}, "
                f"seed={result.config.seed}"
            ),
            f"examples: {result.examples}",
            "",
            "Wall-clock breakdown:",
            f"  self-play/MCTS wall: {breakdown.self_play_seconds:.6f}s",
            (
                "  network inference inside MCTS: "
                f"{breakdown.network_inference_seconds:.6f}s "
                f"({result.network_evals} evals)"
            ),
            f"  self-play MCTS non-inference: {breakdown.mcts_non_inference_seconds:.6f}s",
            (
                "  train step: "
                f"{breakdown.train_step_seconds:.6f}s "
                f"({result.train_steps} steps)"
            ),
            f"  train iteration wall: {breakdown.train_iteration_seconds:.6f}s",
            "",
            "Throughput:",
            (f"  self-play games/sec: {throughput.self_play_games_per_sec:.3f}"),
            (f"  MCTS net-evals/sec: {throughput.mcts_net_evals_per_sec:.3f}"),
            (
                "  network inference evals/sec: "
                f"{throughput.network_inference_evals_per_sec:.3f}"
            ),
            f"  train steps/sec: {throughput.train_steps_per_sec:.3f}",
            "",
            (
                "Dominant cost: "
                f"{dominant_label} ({dominant_seconds:.6f}s, "
                f"{dominant_percent:.1f}% of measured additive time)"
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = BenchmarkConfig(
        game_name=args.game,
        self_play_games=args.self_play_games,
        mcts_simulations=args.mcts_sims,
        mcts_batch_size=args.mcts_batch_size,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        dirichlet_eps=args.dirichlet_eps,
        lr=args.lr,
        l2_reg=args.l2_reg,
        device=args.device,
        torch_threads=args.torch_threads,
    )

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        result = run_benchmark(config)
        profiler.disable()
        print(format_report(result))
        _print_profile(profiler, args.profile_top)
        return 0

    result = run_benchmark(config)
    print(format_report(result))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaZero self-play, MCTS inference, and training."
    )
    parser.add_argument(
        "--game",
        choices=GAME_CHOICES,
        default="tictactoe",
    )
    parser.add_argument("--self-play-games", type=int, default=2)
    parser.add_argument("--mcts-sims", type=int, default=32)
    parser.add_argument("--mcts-batch-size", type=int, default=16)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dirichlet-eps", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=25)
    return parser.parse_args(argv)


def _print_profile(profiler: cProfile.Profile, top_n: int) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(top_n)
    print("")
    print(f"cProfile top {top_n} by cumulative time:")
    print(stream.getvalue(), end="")


def _validate_config(config: BenchmarkConfig) -> None:
    if config.self_play_games <= 0:
        raise ValueError("self_play_games must be positive")
    if config.mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if config.mcts_batch_size <= 0:
        raise ValueError("mcts_batch_size must be positive")
    if config.train_epochs <= 0:
        raise ValueError("train_epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if config.l2_reg < 0:
        raise ValueError("l2_reg must be non-negative")
    if config.torch_threads is not None and config.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")


def _rate(count: int, seconds: float) -> float:
    return count / max(seconds, 1e-12)


if __name__ == "__main__":
    raise SystemExit(main())
