# FSA API Explorer Project Context

## Project Overview
This project is a Streamlit-based web application designed to fetch, analyze, and store food hygiene rating data from the Food Standards Agency (FSA) API. It integrates with Google BigQuery for data persistence, allowing users to maintain a master list of restaurant ratings, identify new establishments, profile restaurant authenticity and value using Gemini models, and update existing records.

## Architecture & Tech Stack
*   **Frontend/UI:** Streamlit (`app/ui/st_app.py`)
*   **Backend Logic:** Python 3.11 / Python 3.13 (`.venv`)
*   **Data Source:** FSA API (UK Food Hygiene Ratings) & Google Places (New) API
*   **Data Storage & ML:** Google BigQuery & BigQuery ML (Boosted Tree Regressor)
*   **AI Models:** `gemini-3.8-flash` (production standard) with ADK Agent integration and Google Maps Grounding
*   **Containerization:** Docker
*   **CI/CD:** Google Cloud Build (`cloudbuild.yaml`) & Cloud Run (`europe-west2`)

## Key Files and Directories

### Source Code (`app/`)
*   **`app/ui/st_app.py`**: The main entry point for the Streamlit application. It handles UI layout, coordinate filtering, Gemini profiling, and BigQuery data orchestration.
*   **`app/services/bq_utils.py`**: Streamlined utility module for all BigQuery interactions (data loading, schema management, merge operations, and Gemini SQL enrichments).
*   **`app/services/api_client.py`**: Contains `fetch_api_data` to interact with the UK Food Standards Agency API.
*   **`app/core/data_processing.py`**: Handles coordinate parsing, API data normalization, duplicate detection, and 6-pillar Gemini structured metric extraction.
*   **`app/agent.py`**: Cloud-native ADK agent definition configured with `gemini-3.8-flash` and `GoogleMapsGroundingTool`.
*   **`app/fast_api_app.py`**: FastAPI wrapper with resilient telemetry and authentication fallbacks for local/CI test isolation and Cloud Run deployment.

### Maintenance & Migration Scripts (`scripts/`)
*   **`scripts/bq_scripts.py`**: SQL templates for Gemini enrichment (`gemini-3.8-flash`) and table merge workflows.
*   **`scripts/enrich_maps_data.py`**: Streamlined Google Places API batch enrichment.
*   **`scripts/train_bqml_model.py`**: BQML Boosted Tree Regressor model training with pre-flight JIT enrichment checks.
*   **`scripts/reconcile_fsa_liveness.py`**: Probes UK FSA `/Establishments/{fhrsid}` liveness (`200` vs `404`), retires delisted rows (`fsa_status = 'delisted'`), links superseded FHRSIDs to active replacements within 100 m (`superseded_by`), and carries over user ratings (newest rating by `first_seen` wins), triage scope, and enrichment metadata.
*   **`scripts/migrate_*.py`**: Concise BigQuery column migration scripts (`migrate_user_rating.py`, `migrate_predicted_rating.py`, `migrate_maps_columns.py`, `migrate_more_maps_columns.py`, `migrate_rating_scale_6pt.py`, `migrate_fsa_status_columns.py`).

### Testing & Quality Flywheel (`tests/` & `app/**/test_*.py`)
*   **Offline Test Suite:** 630 unit and module tests covering UI components, core processing, API clients, BigQuery utilities, weekly liveness reconciliation, and migration scripts.
*   **`tests/` Test Suite:** Integration, evaluation, and BQML regression tests (`test_restaurant_eval.py`, `test_server_e2e.py`, `test_bqml_training.py`, etc.).

### Agent Skills (`_agents/skills/` & `~/.gemini/config/skills/`)
*   **`_agents/skills/restaurant-uniqueness-explorer/SKILL.md`** (also mirrored globally at `~/.gemini/config/skills/restaurant-uniqueness-explorer/SKILL.md`): Parallel multi-agent workflow (1 subagent per shortlisted restaurant) that performs a 30% 6-pillar sanity check and a 70% deep menu & native-language dish archaeology sweep (with concise telegraph-style output) to answer *"Is it worth going?"* and *"What unique regional dishes should I order?"*. Includes `scripts/fetch_bq_shortlist.py` and `references/subagent_prompt_template.md`.

## Setup and Usage

### Prerequisites
*   Python 3.11+
*   Google Cloud SDK (`gcloud`) configured with Application Default Credentials (ADC) for BigQuery access.

### Installation & Environment
The project consolidates all dependencies in a single `.venv` virtual environment:
```bash
source .venv/bin/activate
uv sync
```

### Running Locally
To start the Streamlit app:
```bash
streamlit run app/ui/st_app.py
```
The app will be accessible at `http://localhost:8501`.

### Running Tests & Evaluation
```bash
# The offline suite -- the whole repo, minus the tests marked `integration`.
# This is what Cloud Build runs.
pytest

# The live tests, deliberately. Real Vertex calls; they cost money.
pytest -m integration

# The ADK evaluation flywheel.
adk eval app tests/eval/evalsets/restaurant_eval.evalset.json
```

## Development Conventions

*   **Virtual Environment:** Strictly use `.venv` for all package installations and test executions.
*   **AI Model Standards:** All ADK agents and BigQuery enrichment queries strictly use `gemini-3.8-flash` (or `gemini-3.1-pro`). Legacy models (`gemini-2.5-flash`, `gemini-1.5-*`, `gemini-3.7-flash`) are strictly deprecated.
*   **Code Simplification & Cleanliness:** Avoid redundant boilerplate, verbose duplicate logging, and root-level scratch files. Keep scripts and services modular and concise.
*   **FastAPI & Telemetry Resiliency:** Local test suites execute with `INTEGRATION_TEST=TRUE` to disable live Cloud Trace network roundtrips, while Cloud Run uses full OpenTelemetry tracing.
*   **BigQuery Schemas & Column Sanitization:** Explicit schemas are defined in `bq_utils.py`. Strict column sanitization (`sanitize_column_name`) ensures BigQuery compatibility.
*   **Pipeline Queue Sentinels & Stage-1 Gating Invariants:** Never stamp `predicted_user_rating` on unprofiled rows (`gemini_insights_structured IS NULL`), as a non-null fallback score (`1.00`) hides unprocessed rows from `WHERE predicted_user_rating IS NULL` queues. Untriaged (`in_scope IS NULL`) rows must auto-resolve `in_scope` when `SCRIPT_MERGE_INSIGHTS` merges Pillar 6 (`is_sit_down_restaurant AND type = 'RESTAURANT_DINING'`) and must not be Stage-1 capped prior to profiling.
*   **Grounded `AI.GENERATE` Batching:** Keep grounded `AI.GENERATE` (Google Search + Maps tools) batches at `<= 15` rows per query; for large backfills, fan out `10–15` row chunks into isolated temporary tables and run a single consolidated `MERGE` into `fsa_master` to avoid timeouts and BigQuery DML lock contention.

## Cloud Deployment
The application is deployed to Google Cloud Run via Cloud Build.

> [!IMPORTANT]
> A Cloud Build trigger is configured to automatically build and deploy the application with every commit and push to the repository. **You must commit and push your changes to the remote repository (`main` branch) to see them reflected in the live application.**

## Recent Updates
*   **FSA Liveness Tracking & Duplicate Reconciliation:** Added `last_seen`, `fsa_status`, `delisted_at`, and `superseded_by` to `fsa_master` (`scripts/migrate_fsa_status_columns.py`) and built `scripts/reconcile_fsa_liveness.py` + incremental weekly liveness maintenance in `app/cron/fetch_weekly.py`. Reconciled all 207 name+postcode duplicate rows (`99` active, `108` delisted, `89` linked to active replacements within 100 m), carried user ratings/profiles onto live FHRSIDs (newest rating by `first_seen` wins), hid delisted rows by default in the Streamlit UI (with a `"Show closed / re-listed restaurants"` sidebar toggle and `🔁 superseded by <id>` / `🚫 closed` badges), and excluded delisted rows from scoring/enrichment while keeping their historical ratings in BQML training.
*   **1-to-6 Rating Scale Migration:** Migrated `user_rating` and `predicted_user_rating` (along with `tree_pred`, `lin_pred`, Active Learning VoI thresholds, and UI sliders/inputs) to a 1-to-6 scale (`scripts/migrate_rating_scale_6pt.py`), backed up historical ratings (`user_rating_10pt_backup` and `user_rating_7pt_backup`), and retrained/rescored the BQML hybrid ensemble (`restaurant_preference_model` + `restaurant_preference_model_linear`).
*   **Gemini Model Upgrade to `gemini-3.8-flash`:** Upgraded all ADK restaurant agents, Google Maps grounding profiler, BigQuery `AI.GENERATE` utilities, and evaluation configs to `gemini-3.8-flash`. Added comprehensive test coverage in `tests/test_model_upgrades.py`.
*   **Deep Codebase Refactoring & Bloat Elimination:** Executed an aggressive codebase cleanup across backend modules and root files. Removed the orphaned boilerplate directory `reference_auth/`, scratch scripts (`find_file.sh`, `gemini_config.sh`, `example_results.json`), obsolete ADK wrapper `my_app.py`, and dead service module `agent_orchestrator.py`. Pruned unused BigQuery helper functions (`update_rows_in_bigquery`, `execute_merge_query`, `upsert_agent_insight`, `load_specific_agent_insights`) and their associated dead tests, eliminating ~900+ lines of unused code while maintaining 100% test pass rates across all 74 active tests.
*   **Previous Codebase Simplification:** Streamlined `app/services/bq_utils.py` (-50%), `app/core/data_processing.py` (-40%), and `scripts/enrich_maps_data.py` (-52%).
*   **Model Standardization:** Updated all agent configurations and test assertions to `gemini-3.8-flash`.
*   **FastAPI & Test Suite Hardening:** Added graceful Cloud Logging / Auth fallbacks to `fast_api_app.py`, normalized endpoint URL construction in integration tests, and added BQ mock isolation for deterministic CI testing.