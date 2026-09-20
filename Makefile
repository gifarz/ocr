# Convenience wrapper around scripts/run.sh, pytest, and Docker - every
# target here is just a short name for a command you could type yourself;
# run `make help` (or just `make`) to see them all. Nothing here is
# required - the scripts/Dockerfile these delegate to work fine on their
# own too.

.DEFAULT_GOAL := help
.PHONY: help setup run run-prod test docker-build docker-run compose-up compose-down clean

help:
	@echo "Available targets:"
	@echo "  make setup        - create .venv/ and install requirements.txt into it"
	@echo "  make run          - run locally, dev mode (auto-reload), port 8088"
	@echo "  make run-prod     - run locally, prod mode (no reload), binds 0.0.0.0"
	@echo "  make test         - run the test suite (needs tesseract + .venv, see setup)"
	@echo "  make docker-build - build the service's Docker image"
	@echo "  make docker-run   - run the built image standalone (needs .env - see .env.example)"
	@echo "  make compose-up   - build (if needed) and run via docker compose"
	@echo "  make compose-down - stop the docker compose service"
	@echo "  make clean        - remove .venv/, __pycache__/, .pytest_cache/"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -q --upgrade pip && pip install -q -r requirements.txt
	@echo "Done - now run 'make run' (or ./scripts/run.sh)."

run:
	./scripts/run.sh dev

run-prod:
	./scripts/run.sh prod

test:
	. .venv/bin/activate && pytest tests/ -v

docker-build:
	docker build -t complifi-ocr-service .

docker-run:
	docker run --rm -p 8088:8088 --env-file .env complifi-ocr-service

compose-up:
	docker compose up --build

compose-down:
	docker compose down

clean:
	rm -rf .venv .pytest_cache
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
