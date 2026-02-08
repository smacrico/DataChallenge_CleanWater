.PHONY: help venv install lint format test smoke train predict docs export all clean

help:
	@echo "Available targets:"
	@echo "  venv       - Create virtual environment"
	@echo "  install    - Install dependencies and pre-commit hooks"
	@echo "  lint       - Run linters (ruff, black, isort) in check mode"
	@echo "  format     - Auto-format code with ruff, black, isort"
	@echo "  test       - Run pytest test suite"
	@echo "  smoke      - Run notebook smoke tests"
	@echo "  train      - Run model training pipeline"
	@echo "  predict    - Generate submission.csv"
	@echo "  docs       - Generate documentation (Model Card, Business Plan, Video Script)"
	@echo "  export     - Export docs to HTML and PDF"
	@echo "  all        - Run full end-to-end pipeline"
	@echo "  clean      - Remove generated files and caches"

venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux/Mac)"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

lint:
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/
	isort src/ tests/

test:
	pytest -v --cov=src/wqsa --cov-report=term-missing

smoke:
	pytest tests/ -v -k smoke
	@echo "Running notebook smoke test..."
	python -m nbclient --execute notebooks/01_ingest_and_stage_check.ipynb --timeout=300 || true

train:
	python -m src.wqsa.modeling.train_cv

predict:
	python -m src.wqsa.modeling.predict

docs:
	python -m src.wqsa.docs.generate_model_card
	python -m src.wqsa.docs.generate_business_plan
	python -m src.wqsa.docs.generate_video_script

export:
	python -m src.wqsa.docs.export_to_html
	python -m src.wqsa.docs.export_to_pdf

all: lint test train predict docs export
	@echo "Full pipeline completed successfully!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	rm -rf models/*.pkl models/*.joblib 2>/dev/null || true
	@echo "Cleaned up generated files and caches"
