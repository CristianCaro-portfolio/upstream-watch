.PHONY: install data run smoke test lint ask brief mcp

install:
	pip install -e ".[dev]"

data:
	python -m upstream_watch ingest

run:
	python -m upstream_watch run

smoke:
	UPW_EMBEDDER=hash python -m upstream_watch run --sample --out-dir /tmp/upstream-smoke

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

ask:
	python -m upstream_watch ask "Little Falls just hit 11.7 feet. What is under water?"

brief:
	python -m upstream_watch brief --event 2025-05-15

mcp:
	python -m upstream_watch mcp
