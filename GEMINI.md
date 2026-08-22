# FSA API Explorer Project Context

## Project Overview
This project is a Streamlit-based web application designed to fetch, analyze, and store food hygiene rating data from the Food Standards Agency (FSA) API. It integrates with Google BigQuery for data persistence, allowing users to maintain a master list of restaurant ratings, identify new establishments, profile restaurant authenticity and value using Gemini models, and update existing records.

## Architecture & Tech Stack
*   **Frontend/UI:** Streamlit (`app/ui/st_app.py`)
*   **Backend Logic:** Python 3.11 / Python 3.13 (`.venv`)
*   **Data Source:** FSA API (UK Food Hygiene Ratings) & Google Places (New) API
*   **Data Storage & ML:** Google BigQuery & BigQuery ML (Boosted Tree Regressor)
*   **AI Models:** `gemini-3.7-flash` (production standard) with ADK Agent integration and Google Maps Grounding
*   **Containerization:** Docker
*   **CI/CD:** Google Cloud Build (`cloudbuild.yaml`) & Cloud Run (`europe-west2`)

## Key Files and Directories

### Source Code (`app/`)
*   **`app/ui/st_app.py`**: The main entry point for the Streamlit application. It handles UI layout, coordinate filtering, Gemini profiling, and BigQuery data orchestration.
*   **`app/services/bq_utils.py`**: Streamlined utility module for all BigQuery interactions (data loading, schema management, merge operations, and Gemini SQL enrichments).
*   **`app/services/api_client.py`**: Contains `fetch_api_data` to interact with the UK Food Standards Agency API.
*   **`app/core/data_processing.py`**: Handles coordinate parsing, API data normalization, duplicate detection, and 6-pillar Gemini structured metric extraction.
*   **`app/agent.py` & `app/maps_agent/agent.py`**: Cloud-native ADK agent definitions configured with `gemini-3.7-flash` and `GoogleMapsGroundingTool`.
*   **`app/fast_api_app.py`**: FastAPI wrapper with resilient telemetry and authentication fallbacks for local/CI test isolation and Cloud Run deployment.

### Maintenance & Migration Scripts (`scripts/`)
*   **`scripts/bq_scripts.py`**: SQL templates for Gemini enrichment (`gemini-3.7-flash`) and table merge workflows.
*   **`scripts/enrich_maps_data.py`**: Streamlined Google Places API batch enrichment.
*   **`scripts/train_bqml_model.py`**: BQML Boosted Tree Regressor model training with pre-flight JIT enrichment checks.
*   **`scripts/migrate_*.py`**: Concise BigQuery column migration scripts (`migrate_user_rating.py`, `migrate_predicted_rating.py`, `migrate_maps_columns.py`, `migrate_more_maps_columns.py`).

### Testing & Quality Flywheel (`tests/` & `app/**/test_*.py`)
*   **`app/` Test Suite:** 58 unit and module tests covering UI components, core processing, API clients, and BigQuery utilities (pruned tests for dead modules and obsolete UI tabs).
*   **`tests/` Test Suite:** 16 integration, evaluation, and BQML regression tests (`test_restaurant_eval.py`, `test_server_e2e.py`, `test_bqml_training.py`, etc.).

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
To execute the full test suite and the ADK evaluation flywheel:
```bash
# Run all unit, integration, and eval tests
pytest app/ tests/

# Run ADK agent evaluation dataset
agents-cli eval run --evalset tests/eval/evalsets/restaurant_eval.evalset.json
```

## Development Conventions

*   **Virtual Environment:** Strictly use `.venv` for all package installations and test executions.
*   **AI Model Standards:** All ADK agents and BigQuery enrichment queries strictly use `gemini-3.7-flash` (or `gemini-3.1-pro`). Legacy models (`gemini-2.5-flash`, `gemini-1.5-*`) are strictly deprecated.
*   **Code Simplification & Cleanliness:** Avoid redundant boilerplate, verbose duplicate logging, and root-level scratch files. Keep scripts and services modular and concise.
*   **FastAPI & Telemetry Resiliency:** Local test suites execute with `INTEGRATION_TEST=TRUE` to disable live Cloud Trace network roundtrips, while Cloud Run uses full OpenTelemetry tracing.
*   **BigQuery Schemas & Column Sanitization:** Explicit schemas are defined in `bq_utils.py`. Strict column sanitization (`sanitize_column_name`) ensures BigQuery compatibility.

## Cloud Deployment
The application is deployed to Google Cloud Run via Cloud Build.

> [!IMPORTANT]
> A Cloud Build trigger is configured to automatically build and deploy the application with every commit and push to the repository. **You must commit and push your changes to the remote repository (`main` branch) to see them reflected in the live application.**

## Recent Updates
*   **Deep Codebase Refactoring & Bloat Elimination:** Executed an aggressive codebase cleanup across backend modules and root files. Removed the orphaned boilerplate directory `reference_auth/`, scratch scripts (`find_file.sh`, `gemini_config.sh`, `example_results.json`), obsolete ADK wrapper `my_app.py`, and dead service module `agent_orchestrator.py`. Pruned unused BigQuery helper functions (`update_rows_in_bigquery`, `execute_merge_query`, `upsert_agent_insight`, `load_specific_agent_insights`) and their associated dead tests, eliminating ~900+ lines of unused code while maintaining 100% test pass rates across all 74 active tests.
*   **Previous Codebase Simplification:** Streamlined `app/services/bq_utils.py` (-50%), `app/core/data_processing.py` (-40%), and `scripts/enrich_maps_data.py` (-52%).
*   **Model Standardization:** Updated all agent configurations and test assertions to `gemini-3.7-flash`.
*   **FastAPI & Test Suite Hardening:** Added graceful Cloud Logging / Auth fallbacks to `fast_api_app.py`, normalized endpoint URL construction in integration tests, and added BQ mock isolation for deterministic CI testing.