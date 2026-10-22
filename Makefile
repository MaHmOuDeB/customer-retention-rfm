.PHONY: setup data analysis export test lint all

setup:
	uv sync --locked

data:
	uv run python src/download.py
	uv run python src/prepare.py

analysis:
	uv run python src/analysis.py

export:
	uv run python src/export_site.py

test:
	uv run pytest -q

lint:
	uv run ruff check .

all: data analysis
