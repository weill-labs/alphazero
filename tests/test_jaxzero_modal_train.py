"""Tests for the optional JAX Modal GPU-training wrapper."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_jax_modal_train():
    sys.modules.pop("jaxzero.modal_train", None)
    return importlib.import_module("jaxzero.modal_train")


def test_jaxzero_modal_train_imports_without_modal_installed(monkeypatch) -> None:
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "modal", raising=False)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = load_jax_modal_train()

    assert module.app is None
    assert module.image is None
    assert module.checkpoint_volume is None
    with pytest.raises(RuntimeError, match="uv sync --extra modal"):
        module.train_remote()
    with pytest.raises(RuntimeError, match="uv sync --extra modal"):
        module.checkpoint_elo_remote()
    with pytest.raises(RuntimeError, match="uv sync --extra modal"):
        module.checkpoint_elo()


def test_jaxzero_modal_train_registers_gpu_function(monkeypatch) -> None:
    class FakeImage:
        def __init__(self, python_version: str | None) -> None:
            self.python_version = python_version
            self.packages: tuple[str, ...] = ()
            self.modules: tuple[str, ...] = ()

        def pip_install(self, *packages: str):
            self.packages = packages
            return self

        def add_local_python_source(self, *modules: str):
            self.modules = modules
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(python_version: str | None = None) -> FakeImage:
            return FakeImage(python_version)

    class FakeFunction:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def with_options(self, **kwargs):
            self.options.update(kwargs)
            return self

        def remote(self, **kwargs):
            return kwargs

    class FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name
            self.functions: list[FakeFunction] = []
            self.entrypoint = None

        def function(self, **options):
            def decorate(func):
                function = FakeFunction(options)
                self.functions.append(function)
                return function

            return decorate

        def local_entrypoint(self):
            def decorate(func):
                self.entrypoint = func
                return func

            return decorate

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name)

    class FakeVolume:
        def __init__(self, name: str) -> None:
            self.name = name

        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> "FakeVolume":
            return FakeVolume(name)

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
        Volume=FakeVolume,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    module = load_jax_modal_train()

    assert module.app.name == "jaxzero"
    assert module.image.python_version == "3.12"
    assert module.image.packages == (
        "jax[cuda12]",
        "pgx>=2.6.0",
        "mctx>=0.0.6",
        "flax>=0.12.7",
        "optax>=0.2.8",
        "wandb>=0.27.0",
    )
    assert module.image.modules == ("jaxzero", "alphazero")
    assert len(module.app.functions) == 2
    train_options = module.app.functions[0].options
    assert train_options["image"] is module.image
    assert train_options["gpu"] == "A10G"
    assert train_options["timeout"] == 12 * 60 * 60
    volumes = train_options["volumes"]
    assert volumes["/checkpoints"].name == "alphazero-checkpoints"
    elo_options = module.app.functions[1].options
    assert elo_options["image"] is module.image
    assert elo_options["gpu"] == "A100-40GB"
    assert elo_options["timeout"] == 6 * 60 * 60
    assert "secrets" not in elo_options
    assert elo_options["volumes"]["/checkpoints"].name == "alphazero-checkpoints"
    assert module.app.entrypoint is not None


def test_jaxzero_modal_train_checkpoint_paths(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    module = load_jax_modal_train()

    final_path, checkpoint_dir = module._resolve_checkpoint_paths(
        "connectfour", "run123"
    )

    assert checkpoint_dir == "/checkpoints/run123/connectfour"
    assert final_path == "/checkpoints/run123/connectfour/final.msgpack"
    othello_final, othello_dir = module._resolve_checkpoint_paths("othello", "run123")
    assert othello_dir == "/checkpoints/run123/othello"
    assert othello_final == "/checkpoints/run123/othello/final.msgpack"
    assert module._wandb_project_for_game("connectfour") == "alphazero-connectfour"
    assert module._wandb_project_for_game("othello") == "alphazero-othello"
    assert module._checkpoint_run_tag(SimpleNamespace(id="abc123"), seed=0) == "abc123"
    assert (
        module._checkpoint_run_tag(
            SimpleNamespace(id="abc123"), seed=0, run_tag="othello-resnet-s101"
        )
        == "othello-resnet-s101"
    )
    with pytest.raises(ValueError, match="run_tag"):
        module._checkpoint_run_tag(SimpleNamespace(id="abc123"), seed=0, run_tag="../x")
    assert module._function_call_id(SimpleNamespace(object_id="fc-123")) == "fc-123"
    assert module._function_call_id(SimpleNamespace(id="fc-456")) == "fc-456"
    assert module._resolve_max_steps("connectfour", module._AUTO_MAX_STEPS) == 64
    assert module._resolve_max_steps("othello", module._AUTO_MAX_STEPS) == 128
    assert (
        module._resolve_solver_eval_positions(
            "connectfour", module._AUTO_SOLVER_EVAL_POSITIONS
        )
        == 64
    )
    assert (
        module._resolve_solver_eval_positions(
            "othello", module._AUTO_SOLVER_EVAL_POSITIONS
        )
        == 0
    )
    with pytest.raises(ValueError, match="solver_eval_positions"):
        module._resolve_solver_eval_positions("othello", 16)
    with pytest.raises(ValueError, match="supports games"):
        module._resolve_checkpoint_paths("tictactoe", "run123")
    assert module._split_checkpoint_refs("a.msgpack, b.msgpack\nc.msgpack") == [
        "a.msgpack",
        "b.msgpack",
        "c.msgpack",
    ]
    assert (
        module._checkpoint_volume_path("run/othello/final.msgpack")
        == "/checkpoints/run/othello/final.msgpack"
    )
    assert (
        module._checkpoint_volume_path("/checkpoints/run/othello/final.msgpack")
        == "/checkpoints/run/othello/final.msgpack"
    )
    with pytest.raises(ValueError, match="under /checkpoints"):
        module._checkpoint_volume_path("/tmp/final.msgpack")
    with pytest.raises(ValueError, match="must not contain"):
        module._checkpoint_volume_path("../final.msgpack")


def test_jaxzero_modal_checkpoint_elo_remote_uses_volume_paths(monkeypatch) -> None:
    class FakeImage:
        def pip_install(self, *packages: str):
            return self

        def add_local_python_source(self, *modules: str):
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(python_version: str | None = None) -> FakeImage:
            return FakeImage()

    class FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **options):
            def decorate(func):
                return func

            return decorate

        def local_entrypoint(self):
            def decorate(func):
                return func

            return decorate

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name)

    class FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> "FakeVolume":
            return FakeVolume()

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
        Volume=FakeVolume,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    module = load_jax_modal_train()

    import jaxzero.checkpoint_elo as checkpoint_elo_module

    captured: dict[str, object] = {}

    def fake_resolve_checkpoint_paths(*, checkpoints, checkpoint_dir=None, pattern):
        captured["resolve"] = {
            "checkpoints": list(checkpoints),
            "checkpoint_dir": checkpoint_dir,
            "pattern": pattern,
        }
        return [Path(path) for path in checkpoints]

    def fake_evaluate_checkpoint_ladder(paths, **kwargs):
        captured["evaluate"] = {"paths": list(paths), **kwargs}

        class FakeResult:
            pairings = [SimpleNamespace(games=4), SimpleNamespace(games=4)]

            def as_dict(self):
                return {
                    "game": kwargs["game"],
                    "evaluator_mode": kwargs["evaluator_mode"],
                }

        return FakeResult()

    def fake_trace_checkpoint_game(paths, **kwargs):
        captured["trace"] = {"paths": list(paths), **kwargs}
        return {
            "game": kwargs["game"],
            "evaluator_mode": kwargs["evaluator_mode"],
            "pairing": {"games": kwargs["games"]},
            "summaries": [],
        }

    def fake_probe_checkpoint_state(paths, **kwargs):
        captured["probe"] = {"paths": list(paths), **kwargs}
        return {
            "game": kwargs["game"],
            "games": kwargs["games"],
            "target_ply": kwargs["target_ply"],
            "summaries": [],
            "lanes": [],
        }

    def fake_evaluate_forced_actions(paths, **kwargs):
        captured["force"] = {"paths": list(paths), **kwargs}
        return {
            "game": kwargs["game"],
            "games": kwargs["games"],
            "target_ply": kwargs["target_ply"],
            "actions": [{"action": action} for action in kwargs["force_actions"]],
        }

    monkeypatch.setattr(
        checkpoint_elo_module, "resolve_checkpoint_paths", fake_resolve_checkpoint_paths
    )
    monkeypatch.setattr(
        checkpoint_elo_module,
        "evaluate_checkpoint_ladder",
        fake_evaluate_checkpoint_ladder,
    )
    monkeypatch.setattr(
        checkpoint_elo_module,
        "trace_checkpoint_game",
        fake_trace_checkpoint_game,
    )
    monkeypatch.setattr(
        checkpoint_elo_module,
        "probe_checkpoint_state",
        fake_probe_checkpoint_state,
    )
    monkeypatch.setattr(
        checkpoint_elo_module,
        "evaluate_forced_actions",
        fake_evaluate_forced_actions,
    )

    result = module.checkpoint_elo_remote(
        game="othello",
        checkpoint_paths=[
            "othello-resnet-s102/othello/iter_0080.msgpack",
            "/checkpoints/othello-transformer-s102/othello/iter_0060.msgpack",
        ],
        pattern="iter_*.msgpack",
        mode="round-robin",
        games_per_pairing=4,
        max_steps=module._AUTO_MAX_STEPS,
        evaluator_mode="mcts",
        mcts_simulations=16,
        gumbel_scale=0.0,
        seed=3,
        fit_iterations=20,
        elo_k=8.0,
        requested_gpu="A100-40GB",
    )

    assert captured["resolve"] == {
        "checkpoints": [
            "/checkpoints/othello-resnet-s102/othello/iter_0080.msgpack",
            "/checkpoints/othello-transformer-s102/othello/iter_0060.msgpack",
        ],
        "checkpoint_dir": None,
        "pattern": "iter_*.msgpack",
    }
    assert captured["evaluate"]["game"] == "othello"
    assert captured["evaluate"]["max_steps"] == 128
    assert captured["evaluate"]["mode"] == "round-robin"
    assert captured["evaluate"]["games_per_pairing"] == 4
    assert captured["evaluate"]["evaluator_mode"] == "mcts"
    assert captured["evaluate"]["mcts_simulations"] == 16
    assert captured["evaluate"]["seed"] == 3
    assert captured["evaluate"]["fit_iterations"] == 20
    assert captured["evaluate"]["elo_k"] == 8.0
    assert result["game"] == "othello"
    assert result["evaluator_mode"] == "mcts"
    assert result["checkpoint_volume"] == "alphazero-checkpoints"
    assert result["requested_gpu"] == "A100-40GB"
    assert result["modal_metrics"]["modal_checkpoint_elo_pairings"] == 2
    assert result["modal_metrics"]["modal_checkpoint_elo_games"] == 8

    trace_result = module.checkpoint_elo_remote(
        game="othello",
        checkpoint_paths=[
            "othello-resnet-s102/othello/iter_0080.msgpack",
            "/checkpoints/othello-transformer-s102/othello/iter_0060.msgpack",
        ],
        games_per_pairing=4,
        max_steps=module._AUTO_MAX_STEPS,
        evaluator_mode="mcts",
        mcts_simulations=24,
        trace_plies=3,
        trace_summary_only=True,
    )

    assert captured["trace"]["max_steps"] == 128
    assert captured["trace"]["mcts_simulations"] == 24
    assert captured["trace"]["trace_plies"] == 3
    assert captured["trace"]["summary_only"] is True
    assert trace_result["modal_metrics"]["modal_checkpoint_elo_pairings"] == 1
    assert trace_result["modal_metrics"]["modal_checkpoint_elo_games"] == 4

    probe_result = module.checkpoint_elo_remote(
        game="othello",
        checkpoint_paths=[
            "othello-resnet-s102/othello/iter_0080.msgpack",
            "/checkpoints/othello-transformer-s102/othello/iter_0060.msgpack",
        ],
        games_per_pairing=4,
        max_steps=module._AUTO_MAX_STEPS,
        mcts_simulations=24,
        probe_ply=10,
        probe_budgets="24,32,64",
        probe_top_k=4,
    )

    assert captured["probe"]["max_steps"] == 128
    assert captured["probe"]["replay_simulations"] == 24
    assert captured["probe"]["probe_simulations"] == [24, 32, 64]
    assert captured["probe"]["target_ply"] == 10
    assert captured["probe"]["top_k"] == 4
    assert probe_result["modal_metrics"]["modal_checkpoint_elo_pairings"] == 1
    assert probe_result["modal_metrics"]["modal_checkpoint_elo_games"] == 4

    force_result = module.checkpoint_elo_remote(
        game="othello",
        checkpoint_paths=[
            "othello-resnet-s102/othello/iter_0080.msgpack",
            "/checkpoints/othello-transformer-s102/othello/iter_0060.msgpack",
        ],
        games_per_pairing=4,
        max_steps=module._AUTO_MAX_STEPS,
        mcts_simulations=24,
        force_ply=10,
        force_actions="38,46",
        force_actor="iter_0080",
        continuation_simulations=256,
    )

    assert captured["force"]["max_steps"] == 128
    assert captured["force"]["replay_simulations"] == 24
    assert captured["force"]["continuation_simulations"] == 256
    assert captured["force"]["force_actions"] == [38, 46]
    assert captured["force"]["force_actor"] == "iter_0080"
    assert captured["force"]["target_ply"] == 10
    assert force_result["modal_metrics"]["modal_checkpoint_elo_pairings"] == 1
    assert force_result["modal_metrics"]["modal_checkpoint_elo_games"] == 8


def test_jaxzero_modal_remote_runs_training_and_commits_volume(monkeypatch) -> None:
    class FakeImage:
        def pip_install(self, *packages: str):
            return self

        def add_local_python_source(self, *modules: str):
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(python_version: str | None = None) -> FakeImage:
            return FakeImage()

    class FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **options):
            def decorate(func):
                return func

            return decorate

        def local_entrypoint(self):
            def decorate(func):
                return func

            return decorate

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name)

    class FakeVolume:
        def __init__(self, name: str) -> None:
            self.name = name
            self.committed = False

        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> "FakeVolume":
            return FakeVolume(name)

        def commit(self) -> None:
            self.committed = True

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
        Volume=FakeVolume,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    module = load_jax_modal_train()

    class FakeRun:
        id = "wandb123"
        url = "https://wandb.example/runs/wandb123"

        def __init__(self) -> None:
            self.logs: list[tuple[dict[str, int | float], int]] = []
            self.finished = False

        def log(self, metrics, *, step: int) -> None:
            self.logs.append((dict(metrics), step))

        def finish(self) -> None:
            self.finished = True

    fake_run = FakeRun()
    init_kwargs: dict[str, object] = {}

    def fake_wandb_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    class FakeTrainingConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    captured: dict[str, object] = {}

    def fake_run_training(
        config, *, on_iteration=None, on_checkpoint=None, extra_evaluator=None
    ):
        captured["config"] = config
        captured["extra_evaluator"] = extra_evaluator
        metrics = [
            {"iteration": 0, "loss": 1.25},
            {"iteration": 1, "loss": 0.5},
        ]
        if on_iteration is not None:
            for entry in metrics:
                on_iteration(entry)
        return SimpleNamespace(
            metrics=metrics,
            checkpoint_path=config.checkpoint_path,
        )

    import jaxzero.train as train_module

    monkeypatch.setattr(module, "_wandb_init", fake_wandb_init)
    monkeypatch.setattr(train_module, "TrainingConfig", FakeTrainingConfig)
    monkeypatch.setattr(train_module, "run_training", fake_run_training)

    result = module.train_remote(
        iterations=2,
        batch_size=4,
        num_simulations=5,
        max_steps=6,
        selfplay_temperature=1.0,
        selfplay_temperature_drop_step=8,
        selfplay_temperature_after_drop=0.0,
        selfplay_dirichlet_fraction=0.25,
        selfplay_dirichlet_fraction_drop_step=8,
        selfplay_dirichlet_fraction_after_drop=0.0,
        selfplay_dirichlet_alpha=0.3,
        channels=7,
        num_res_blocks=1,
        learning_rate=0.02,
        solver_eval_positions=0,  # disable inline solver cert for this test
        gating_interval=3,
        gating_games=4,
        gating_threshold=0.6,
        solver_rehearsal_positions=8,
        solver_rehearsal_batch_size=4,
        solver_rehearsal_interval=2,
        solver_rehearsal_seed=123,
        solver_rehearsal_target="wdl",
        solver_rehearsal_solver_max_nodes=10_000,
        solver_rehearsal_policy_loss_weight=1.5,
        solver_rehearsal_value_loss_weight=0.0,
        solver_rehearsal_hard_checkpoint="/checkpoints/ref/connectfour/iter_0050.msgpack",
        solver_rehearsal_hard_pool_size=32,
        solver_rehearsal_hard_sims=64,
        solver_rehearsal_anchor_positions=7,
        seed=9,
        requested_gpu="A100-40GB",
        run_tag="othello-resnet-s9",
    )

    config = captured["config"]
    assert (
        config.checkpoint_path
        == "/checkpoints/othello-resnet-s9/connectfour/final.msgpack"
    )
    assert config.game == "connectfour"
    assert config.iterations == 2
    assert config.batch_size == 4
    assert config.num_simulations == 5
    assert config.max_steps == 6
    assert config.selfplay_temperature == 1.0
    assert config.selfplay_temperature_drop_step == 8
    assert config.selfplay_temperature_after_drop == 0.0
    assert config.selfplay_dirichlet_fraction == 0.25
    assert config.selfplay_dirichlet_fraction_drop_step == 8
    assert config.selfplay_dirichlet_fraction_after_drop == 0.0
    assert config.selfplay_dirichlet_alpha == 0.3
    assert config.arch == "resnet"
    assert config.use_value_cls_token is False
    assert config.policy_head_style == "flatten"
    assert config.input_embed_style == "linear"
    assert config.gating_interval == 3
    assert config.gating_games == 4
    assert config.gating_threshold == 0.6
    assert config.solver_rehearsal_positions == 8
    assert config.solver_rehearsal_batch_size == 4
    assert config.solver_rehearsal_interval == 2
    assert config.solver_rehearsal_seed == 123
    assert config.solver_rehearsal_target == "wdl"
    assert config.solver_rehearsal_solver_max_nodes == 10_000
    assert config.solver_rehearsal_policy_loss_weight == 1.5
    assert config.solver_rehearsal_value_loss_weight == 0.0
    assert (
        config.solver_rehearsal_hard_checkpoint
        == "/checkpoints/ref/connectfour/iter_0050.msgpack"
    )
    assert config.solver_rehearsal_hard_pool_size == 32
    assert config.solver_rehearsal_hard_sims == 64
    assert config.solver_rehearsal_anchor_positions == 7
    assert init_kwargs["project"] == "alphazero-connectfour"
    assert init_kwargs["run_name"] == "jaxzero-modal-connectfour-othello-resnet-s9"
    assert result["checkpoint_path"] == config.checkpoint_path
    assert result["checkpoint_dir"] == "/checkpoints/othello-resnet-s9/connectfour"
    assert result["checkpoint_volume"] == "alphazero-checkpoints"
    assert result["final_metrics"] == {"iteration": 1, "loss": 0.5}
    assert result["config"]["game"] == "connectfour"
    assert result["config"]["requested_gpu"] == "A100-40GB"
    assert result["config"]["run_tag"] == "othello-resnet-s9"
    assert result["config"]["max_steps"] == 6
    assert result["config"]["arch"] == "resnet"
    assert result["config"]["use_value_cls_token"] is False
    assert result["config"]["policy_head_style"] == "flatten"
    assert result["config"]["input_embed_style"] == "linear"
    assert result["config"]["solver_eval_positions"] == 0
    assert result["config"]["solver_rehearsal_positions"] == 8
    assert result["config"]["solver_rehearsal_value_loss_weight"] == 0.0
    assert result["config"]["solver_rehearsal_anchor_positions"] == 7
    assert result["config"]["selfplay_temperature_after_drop"] == 0.0
    assert fake_run.logs[0] == ({"iteration": 0, "loss": 1.25}, 0)
    assert fake_run.logs[1] == ({"iteration": 1, "loss": 0.5}, 1)
    assert fake_run.logs[2][0]["checkpoint_written"] == 1
    assert fake_run.logs[2][1] == 2
    assert fake_run.finished
    assert module.checkpoint_volume.committed
    assert captured["extra_evaluator"] is None  # default off when positions == 0

    othello_result = module.train_remote(
        game="othello",
        iterations=1,
        batch_size=1,
        num_simulations=1,
        max_steps=module._AUTO_MAX_STEPS,
        solver_eval_positions=module._AUTO_SOLVER_EVAL_POSITIONS,
        seed=10,
        run_tag="othello-default-s10",
    )
    othello_config = captured["config"]
    assert othello_config.game == "othello"
    assert othello_config.max_steps == 128
    assert othello_config.arch == "transformer"
    assert othello_config.use_value_cls_token is True
    assert othello_config.policy_head_style == "flatten"
    assert othello_config.input_embed_style == "conv3x3"
    assert (
        othello_config.checkpoint_path
        == "/checkpoints/othello-default-s10/othello/final.msgpack"
    )
    assert othello_result["config"]["game"] == "othello"
    assert othello_result["config"]["solver_eval_positions"] == 0
    assert othello_result["config"]["arch"] == "transformer"
    assert othello_result["config"]["use_value_cls_token"] is True
    assert othello_result["config"]["policy_head_style"] == "flatten"
    assert othello_result["config"]["input_embed_style"] == "conv3x3"
