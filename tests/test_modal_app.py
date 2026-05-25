"""Tests for the optional Modal cloud-training wrapper."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace


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

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    module = load_modal_app()

    assert module.app.name == "alphazero-tictactoe"
    assert module.image.python_version == "3.12"
    assert module.image.packages == ("torch>=2.2", "numpy>=1.26", "wandb>=0.27.0")
    assert module.image.modules == ("alphazero",)
    assert module.app.functions
    assert module.app.functions[0].options["image"] is module.image
    assert module.app.functions[0].options["timeout"] == 6 * 60 * 60
    assert module.app.entrypoint is not None
