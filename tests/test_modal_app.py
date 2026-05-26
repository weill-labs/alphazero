"""Tests for the optional Modal cloud-training wrapper."""

from __future__ import annotations

import builtins
import json
import sys
from types import SimpleNamespace

import pytest


def load_modal_app():
    sys.modules.pop("modal_app", None)
    return __import__("modal_app")


def test_modal_app_imports_without_modal_installed(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = load_modal_app()

    assert module.app is None
    assert module.image is None


def test_modal_app_registers_remote_training_without_real_modal(monkeypatch) -> None:
    class FakeImage:
        def __init__(self, python_version: str | None) -> None:
            self.python_version = python_version
            self.packages: tuple[str, ...] = ()
            self.pip_kwargs: dict[str, object] = {}
            self.modules: tuple[str, ...] = ()

        def pip_install(self, *packages: str, **kwargs):
            self.packages = packages
            self.pip_kwargs = kwargs
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

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    module = load_modal_app()

    assert module.app.name == "alphazero"
    assert module.image.python_version == "3.12"
    assert module.image.packages == ("torch>=2.2", "numpy>=1.26", "wandb>=0.27.0")
    assert module.image.pip_kwargs == {
        "extra_index_url": "https://download.pytorch.org/whl/cpu"
    }
    assert module.image.modules == ("alphazero",)
    assert module.app.functions
    assert module.app.functions[0].options["image"] is module.image
    assert module.app.functions[0].options["timeout"] == 6 * 60 * 60
    assert module.app.entrypoint is not None
    assert module.app.entrypoint.__defaults__[:4] == ("tictactoe", None, None, None)


def test_modal_app_game_defaults_are_game_specific(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = load_modal_app()

    assert module._resolve_training_args(
        game="tictactoe",
        iterations=None,
        self_play_games=None,
        sims=None,
    ) == (60, 24, 128)
    assert module._resolve_training_args(
        game="connectfour",
        iterations=None,
        self_play_games=None,
        sims=None,
    ) == (120, 48, 256)
    assert module._resolve_training_args(
        game="connectfour",
        iterations=3,
        self_play_games=4,
        sims=5,
    ) == (3, 4, 5)
    with pytest.raises(ValueError, match="game must be"):
        module._resolve_training_args(
            game="chess",
            iterations=None,
            self_play_games=None,
            sims=None,
        )


def test_modal_app_eval_args_are_parsed_for_training(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = load_modal_app()

    assert module._resolve_eval_args(
        gating_interval=2,
        gating_games=3,
        gating_threshold=0.75,
        eval_interval=4,
        ladder_games=5,
        ladder_depths="1,3,5",
    ) == {
        "eval_interval": 4,
        "gating_games": 3,
        "gating_interval": 2,
        "gating_threshold": 0.75,
        "ladder_depths": (1, 3, 5),
        "ladder_games": 5,
    }


def test_modal_app_wandb_project_is_game_aware(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = load_modal_app()

    assert module._wandb_project_for_game("tictactoe") == "alphazero-tictactoe"
    assert module._wandb_project_for_game("connectfour") == "alphazero-connectfour"


def test_modal_remote_threads_eval_args_to_training(monkeypatch) -> None:
    class FakeImage:
        def pip_install(self, *packages: str, **kwargs):
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

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    module = load_modal_app()

    captured_kwargs: dict[str, object] = {}

    def fake_train_tictactoe_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return object(), {}

    def fake_play_match(*args, **kwargs):
        return (1, 0, 0)

    import alphazero.arena as arena

    monkeypatch.setattr(module, "_wandb_init", lambda **kwargs: None)
    monkeypatch.setattr(arena, "train_tictactoe_agent", fake_train_tictactoe_agent)
    monkeypatch.setattr(arena, "play_match", fake_play_match)

    result = module.train_remote(
        game="tictactoe",
        iterations=1,
        self_play_games=1,
        sims=1,
        mcts_batch_size=8,
        eval_games=1,
        eval_sims=1,
        gating_interval=2,
        gating_games=3,
        gating_threshold=0.7,
        eval_interval=4,
        ladder_games=5,
        ladder_depths="1,4",
    )

    assert captured_kwargs["gating_interval"] == 2
    assert captured_kwargs["gating_games"] == 3
    assert captured_kwargs["gating_threshold"] == 0.7
    assert captured_kwargs["eval_interval"] == 4
    assert captured_kwargs["ladder_games"] == 5
    assert captured_kwargs["ladder_depths"] == (1, 4)
    assert captured_kwargs["self_play_mcts_cfg"]["batch_size"] == 8
    assert result["config"]["ladder_depths"] == (1, 4)
    assert result["config"]["mcts_batch_size"] == 8


def test_modal_entrypoint_forwards_game_to_remote(monkeypatch, capsys) -> None:
    class FakeImage:
        def pip_install(self, *packages: str, **kwargs):
            return self

        def add_local_python_source(self, *modules: str):
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(python_version: str | None = None) -> FakeImage:
            return FakeImage()

    class FakeFunction:
        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def with_options(self, **kwargs):
            self.options.update(kwargs)
            return self

        def remote(self, **kwargs):
            return kwargs

    class FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name
            self.entrypoint = None

        def function(self, **options):
            def decorate(func):
                return FakeFunction()

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

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    module = load_modal_app()

    module.app.entrypoint(
        game="connectfour",
        iterations=3,
        self_play_games=4,
        sims=5,
        mcts_batch_size=13,
        seed=6,
        gpu="A10G",
        eval_games=7,
        eval_sims=8,
        gating_interval=9,
        gating_games=10,
        gating_threshold=0.65,
        eval_interval=11,
        ladder_games=12,
        ladder_depths="1,2",
    )

    expected = {
        "eval_interval": 11,
        "eval_games": 7,
        "eval_sims": 8,
        "game": "connectfour",
        "gating_games": 10,
        "gating_interval": 9,
        "gating_threshold": 0.65,
        "gpu": "A10G",
        "iterations": 3,
        "ladder_depths": "1,2",
        "ladder_games": 12,
        "mcts_batch_size": 13,
        "seed": 6,
        "self_play_games": 4,
        "sims": 5,
    }
    assert (
        capsys.readouterr().out == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
