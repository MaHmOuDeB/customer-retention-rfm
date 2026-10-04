.PHONY: setup data analysis test lint all

setup:
	python -m pip install -r requirements.txt

data:
	python src/download.py
	python src/prepare.py

analysis:
	python src/analysis.py

test:
	python -m pytest -q

lint:
	python -m ruff check .

all: data analysis
