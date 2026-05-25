"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_wandb_network(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
