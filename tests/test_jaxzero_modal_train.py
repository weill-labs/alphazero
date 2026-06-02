"""Tests for the optional JAX Modal GPU-training wrapper."""

from __future__ import annotations

import builtins
import importlib
import sys
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
    assert module.app.functions
    options = module.app.functions[0].options
    assert options["image"] is module.image
    assert options["gpu"] == "A10G"
    assert options["timeout"] == 12 * 60 * 60
    volumes = options["volumes"]
    assert volumes["/checkpoints"].name == "alphazero-checkpoints"
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
