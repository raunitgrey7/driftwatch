.PHONY: setup lint format test bench bench-quick serve replay docker-build clean

UV ?= uv
PORT ?= 8774

setup:
	$(UV) sync --dev

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run python scripts/check_text.py

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest -q

bench:
	$(UV) run python -m driftwatch.bench

bench-quick:
	$(UV) run python -m driftwatch.bench --quick --out benchmarks/.cache --no-check

serve:
	$(UV) run driftwatch serve --host 127.0.0.1 --port $(PORT)

replay:
	$(UV) run driftwatch replay --api http://127.0.0.1:$(PORT) --scenario covariate_abrupt --batches 180

docker-build:
	docker build -t driftwatch:local .

clean:
	rm -rf artifacts data .pytest_cache .ruff_cache .hypothesis benchmarks/.cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
