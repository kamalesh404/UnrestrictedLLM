.PHONY: install dev test lint check clean run serve pull

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

check: lint typecheck test

run:
	unrestricted run

serve:
	unrestricted serve --port 8080

pull:
	unrestricted pull mistral-7b-instruct

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
