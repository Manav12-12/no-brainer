PYTHON := .venv/bin/python
UV := uv
export PYTHONPATH := src

.PHONY: setup lint typecheck test audit smoke live jev-check jev-plan jev-record experiments fullbrain public-data jev-calibration-plan jev-calibrate

setup:
	$(UV) venv --python 3.11 --allow-existing
	$(UV) pip install --python $(PYTHON) --no-index --find-links wheelhouse -r requirements.lock
	$(PYTHON) scripts/verify_manifest.py
	$(PYTHON) -m sentinel.cli smoke

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest -m "not slow and not fullbrain and not live_jev" --disable-socket --allow-unix-socket --cov --cov-report=term-missing

smoke:
	$(PYTHON) -m sentinel.cli smoke

live:
	$(PYTHON) scripts/run_live_simulation.py

jev-check:
	$(PYTHON) -m sentinel.cli jev-check --confirm-live

jev-plan:
	$(PYTHON) scripts/estimate_jev_calls.py

jev-record:
	$(PYTHON) scripts/run_experiments.py --mode record

experiments:
	$(PYTHON) scripts/run_experiments.py --mode replay

fullbrain:
	$(PYTHON) scripts/run_fullbrain.py

public-data:
	$(PYTHON) scripts/prepare_unsw.py
	$(PYTHON) scripts/run_public_evaluation.py

jev-calibration-plan:
	$(PYTHON) scripts/run_jev_calibration.py --mode plan

jev-calibrate:
	$(PYTHON) scripts/run_jev_calibration.py --mode record --confirm-live

audit: lint typecheck test
	$(PYTHON) -m bandit -c pyproject.toml -r src
	$(PYTHON) -m detect_secrets scan --baseline .secrets.baseline
