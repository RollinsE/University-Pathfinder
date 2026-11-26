.PHONY: install install-colab doctor lint format test test-js demo serve-demo run figures serve notebook clean

install:
	pip install -r requirements-dev.txt

install-colab:
	pip install -r requirements-colab.txt

doctor:
	python scripts/doctor.py

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	python -m pytest

test-js:
	node --test "webapp/tests/*.test.mjs"

# Whole pipeline on synthetic data: no account, a few minutes on a CPU.
# Its web app (with demo models) is written to runs/demo_webapp, never to webapp/.
demo:
	python main.py --mode all --demo

serve-demo:
	python -m http.server 8000 --directory runs/demo_webapp

# Whole pipeline on the real dataset (needs ROBOFLOW_API_KEY; a GPU is strongly recommended).
run:
	python main.py --mode all --figures_dir docs/figures

# Results table and figures for a finished run: make figures RUN=runs/run_YYYYMMDD_HHMMSS
figures:
	python main.py --mode evaluate --run_dir $(RUN) --figures_dir docs/figures

# Serve the web app at http://localhost:8000
serve:
	python -m http.server 8000 --directory webapp

notebook:
	python scripts/build_notebook.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
