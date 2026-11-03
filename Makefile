.PHONY: install data run smoke test lint ask brief mcp data-co run-co brief-co ask-co

install:
	pip install -e ".[dev,colombia]"

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

# Colombia, rio Cauca (IDEAM, datos.gov.co), answers in Spanish
data-co:
	pip install -e ".[colombia]"
	python -m upstream_watch --region cauca ingest

run-co:
	python -m upstream_watch --region cauca run

brief-co:
	python -m upstream_watch --region cauca brief

ask-co:
	python -m upstream_watch --region cauca ask "Puerto Valdivia marca 6 metros, como esta?"
