## Observation
- Consolidated all project virtual environments (`venv`, `bq_venv`) into a single canonical `.venv` environment to eliminate developer and CI tooling confusion.
- Synchronized `pyproject.toml` dependencies so `uv sync` manages all packages in `.venv`.
- Implemented the `agents-cli` Quality Flywheel and Observability framework, establishing `.agents-cli-spec.md`, canonical eval sets in `tests/eval/evalsets/restaurant_eval.evalset.json`, and hybrid evaluation metrics in `tests/eval/eval_config.yaml`.
- Upgraded ADK agent models to `gemini-3.7-flash` in `app/agent.py` and `app/maps_agent/agent.py`.
- Independent execution of `pytest` and `agents-cli eval` in `.venv` passed with a 100% success rate.

## Verification Method
1. Activate the consolidated virtual environment: `source .venv/bin/activate`
2. Run pytest suite: `pytest`
3. Run ADK agent evaluations: `agents-cli eval run --evalset tests/eval/evalsets/restaurant_eval.evalset.json`
