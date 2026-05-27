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
    assert module.app.functions[0].options["cpu"] == 8
    assert module.app.functions[0].options["timeout"] == 6 * 60 * 60
    volumes = module.app.functions[0].options["volumes"]
    assert "/checkpoints" in volumes
    assert volumes["/checkpoints"].name == "alphazero-checkpoints"
    assert module.app.entrypoint is not None
    assert module.app.entrypoint.__defaults__[:4] == ("tictactoe", None, None, None)


def test_modal_app_constructs_with_real_modal() -> None:
    # Smoke test against the real modal package (skipped when it isn't
    # installed). Guards the #7-style regression where modal_app.py was emptied:
    # the fake-modal tests above cannot catch a broken real-modal app/image.
    pytest.importorskip("modal")

    module = load_modal_app()

    assert module.app is not None, "modal_app.app is None — entrypoint missing"
    assert module.app.name == "alphazero"
    assert module.image is not None, "modal_app.image is None"
    # Under real modal these are wrapped objects (a Function / local entrypoint),
    # not plain callables; their existence proves the app was constructed.
    assert module.train_remote is not None
    assert module.main is not None


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
        game="gomoku",
        iterations=None,
        self_play_games=None,
        sims=None,
    ) == (40, 16, 96)
    assert module._resolve_training_args(
        game="go",
        iterations=None,
        self_play_games=None,
        sims=None,
    ) == (40, 16, 96)
    assert module._resolve_training_args(
        game="connectfour",
        iterations=3,
        self_play_games=4,
        sims=5,
    ) == (3, 4, 5)
    with pytest.raises(ValueError, match="unknown game"):
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
        self_play_workers=2,
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
    assert captured_kwargs["n_selfplay_workers"] == 2
    assert result["config"]["ladder_depths"] == (1, 4)
    assert result["config"]["mcts_batch_size"] == 8
    assert result["config"]["self_play_workers"] == 2
    # The trained net is persisted to the mounted volume, and the volume is
    # committed so the write survives container teardown.
    assert captured_kwargs["checkpoint_path"].startswith("/checkpoints/")
    assert captured_kwargs["checkpoint_path"].endswith("/tictactoe/final.pt")
    assert captured_kwargs["checkpoint_every"] is None
    assert captured_kwargs["checkpoint_dir"].startswith("/checkpoints/")
    assert module.checkpoint_volume.committed is True


def test_modal_remote_trains_go_generically(monkeypatch) -> None:
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
            return lambda func: func

        def local_entrypoint(self):
            return lambda func: func

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

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=FakeApp,
            Image=FakeImageFactory,
            Secret=FakeSecret,
            Volume=FakeVolume,
        ),
    )
    module = load_modal_app()

    captured: dict[str, object] = {}

    def fake_train_agent(game, **kwargs):
        captured["game_type"] = type(game).__name__
        captured["ladder_depths"] = kwargs["ladder_depths"]
        return object(), {}

    import alphazero.arena as arena

    monkeypatch.setattr(module, "_wandb_init", lambda **kwargs: None)
    monkeypatch.setattr(arena, "train_agent", fake_train_agent)
    monkeypatch.setattr(arena, "play_match", lambda *a, **k: (1, 0, 0))

    result = module.train_remote(
        game="go", iterations=1, self_play_games=1, sims=1, eval_games=2
    )

    assert captured["game_type"] == "Go"
    assert captured["ladder_depths"] == (1,)  # shallow ladder default for go
    assert "vs_random" in result
    assert "vs_perfect" not in result  # no tractable perfect player for go
    assert result["vs_random"]["win_rate"] == 0.5


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
    module = load_modal_app()

    module.app.entrypoint(
        game="connectfour",
        iterations=3,
        self_play_games=4,
        sims=5,
        mcts_batch_size=13,
        self_play_workers=3,
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
        "checkpoint_every": None,
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
        "self_play_workers": 3,
        "sims": 5,
    }
    assert (
        capsys.readouterr().out == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )


def _load_modal_app_without_modal(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return load_modal_app()


def test_resolve_checkpoint_paths_under_volume(monkeypatch) -> None:
    module = _load_modal_app_without_modal(monkeypatch)
    final_path, checkpoint_dir = module._resolve_checkpoint_paths(
        "connectfour", "7jlfmjgu"
    )
    assert checkpoint_dir == "/checkpoints/7jlfmjgu"
    assert final_path == "/checkpoints/7jlfmjgu/connectfour/final.pt"
    # Final net lives in the same per-game dir as the periodic iter_*.pt files
    # train_agent writes from checkpoint_dir, so a whole run is co-located.
    assert final_path.startswith(checkpoint_dir + "/connectfour/")


def test_checkpoint_run_tag_prefers_wandb_id(monkeypatch) -> None:
    module = _load_modal_app_without_modal(monkeypatch)
    assert module._checkpoint_run_tag(SimpleNamespace(id="abc123"), seed=0) == "abc123"


def test_checkpoint_run_tag_falls_back_without_run(monkeypatch) -> None:
    module = _load_modal_app_without_modal(monkeypatch)
    assert module._checkpoint_run_tag(None, seed=7).startswith("seed7-")
