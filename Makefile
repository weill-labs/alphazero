.PHONY: test modal sync bench

test:
	uv run --extra dev pytest

modal:
	uv run --extra modal modal run modal_app.py

sync:
	uv sync --all-extras

bench:
	uv run --extra dev python -m alphazero.benchmark
