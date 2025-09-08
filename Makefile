.PHONY: help install install-dev test lint format clean build run-docker docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in production mode
	pip install -e .

install-dev: ## Install the package in development mode with all dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests with coverage
	pytest

test-fast: ## Run fast tests only (exclude slow tests)
	pytest -m "not slow"

lint: ## Run linting checks
	ruff check src tests
	mypy src

format: ## Format code with black and ruff
	black src tests
	ruff check --fix src tests

clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

run-docker: ## Build and run Docker container
	docker build -t autotab-ml-agent -f src/service/docker/Dockerfile .
	docker run -p 8000:8000 autotab-ml-agent

docs: ## Build documentation
	cd docs && make html

smoke-test: ## Run smoke test with sample data
	python -m src.cli --data examples/datasets/iris_sample.csv --target species --metric accuracy --budget-minutes 2 --seed 42

# Development workflow shortcuts
check: lint test ## Run all checks (lint + test)

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Data management
setup-data: ## Create data directories and sample files
	mkdir -p data/raw data/artifacts meta models
	touch data/raw/.gitkeep data/artifacts/.gitkeep meta/.gitkeep models/.gitkeep

# Experiment tracking
mlflow-ui: ## Start MLflow UI
	mlflow ui --backend-store-uri sqlite:///meta/mlflow.db --default-artifact-root ./models

# Service management
start-service: ## Start FastAPI service
	uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload

# Git workflow helpers
branch-feat: ## Create a new feature branch (usage: make branch-feat NAME=my-feature)
	@if [ -z "$(NAME)" ]; then echo "Usage: make branch-feat NAME=my-feature"; exit 1; fi
	git checkout develop
	git pull origin develop
	git checkout -b feat/$(NAME)

branch-exp: ## Create a new experiment branch (usage: make branch-exp NAME=dataset-algo-id)
	@if [ -z "$(NAME)" ]; then echo "Usage: make branch-exp NAME=dataset-algo-id"; exit 1; fi
	git checkout develop
	git pull origin develop
	git checkout -b exp/$(NAME)

# Release management
release: ## Create a new release (usage: make release VERSION=0.1.0)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make release VERSION=0.1.0"; exit 1; fi
	git checkout -b release/$(VERSION)
	@echo "Update version in pyproject.toml to $(VERSION)"
	@echo "Update CHANGELOG.md"
	@echo "Commit changes and create PR to main"
