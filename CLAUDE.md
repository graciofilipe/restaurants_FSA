# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All tooling lives in the single canonical `.venv` (managed by `uv`). Never create a second virtualenv.

```bash
source .venv/bin/activate && uv sync    # setup / re-sync deps from pyproject.toml

streamlit run app/ui/st_app.py          # main app, http://localhost:8501

pytest app/ scripts/                    # 296 offline unit tests — this is what Cloud Build runs
pytest app/core/test_scoring_priority.py::test_extract_outcode   # single test
pytest tests/                           # NOT offline-safe (see below)

uvx ruff check .                        # ruff is configured in pyproject.toml but not installed in .venv
```

`tests/` has no skip markers and requires live GCP + network: it makes real Vertex/Gemini calls
(`tests/test_model_upgrades.py::test_live_...`, `tests/eval/`), spins up a uvicorn server on port 8000
(`tests/integration/test_server_e2e.py`), and BQML tests mock BigQuery but the eval tests do not.
Run `pytest app/ scripts/` for the normal edit-test loop.

Operational scripts (all accept `--project_id/--dataset_id`, default to the live project):

```bash
python -m scripts.train_bqml_model --dry-run      # validate BQML training SQL without spending
python -m scripts.train_bqml_model --run_async    # kick off Boosted Tree training
python -m scripts.enrich_maps_data                # Google Places backfill (needs GOOGLE_MAPS_API_KEY)
python -m scripts.enrich_postcode_demographics    # postcodes.io → uk_postcode_demographics
python -m app.cron.fetch_weekly                   # the weekly FSA ingest, run as a Cloud Run Job
```

Deployment is automatic: **a Cloud Build trigger builds and deploys on every push to `main`**. Local
changes are not live until pushed. Manual: `gcloud builds submit --config cloudbuild.yaml .`

## Architecture

A personal restaurant-discovery pipeline: pull new UK restaurant openings from the Food Standards
Agency API, enrich them, score them against one specific user's taste profile, and surface them in a
Streamlit triage UI whose human ratings train the model.

### BigQuery is the entire application state

One table — `filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master` — holds everything:
raw FSA fields, Maps enrichment, Gemini insight JSON, ML predictions, and human labels. There is no
other persistence (ADK sessions are in-memory). Its schema is `MASTER_BQ_SCHEMA` in
`app/services/bq_utils.py`; **all column names are lowercase**, normalized through
`sanitize_column_name`, while the FSA API returns PascalCase — code that straddles the boundary
(`data_processing.py`, `fetch_weekly.py`) checks both spellings. Supporting tables:
`uk_postcode_demographics` (reference), `config_search_params` (cron search coordinates),
`recents_<runid>`/`genairesults_temp_<runid>` (scratch, created and dropped per enrichment run, with
a one-day expiry as a backstop).

### The pipeline

1. **Ingest** — `app/cron/fetch_weekly.py` reads search coordinates from `config_search_params`,
   pages the FSA API (`app/services/api_client.py`), and `process_and_update_master_data` dedupes by
   FHRSID against the IDs already in the master table (`load_fhrsids_from_bq`), appending only
   genuinely new rows with `first_seen`.
2. **Maps enrichment** — `scripts/enrich_maps_data.py` hits Places `searchText` and MERGEs rating,
   review count, price level, coordinates, and types back. Every lookup stamps `maps_lookup_at` and
   `maps_found`; a miss records `maps_found = FALSE` with a NULL rating, and the timestamp — not the
   old `-1` sentinel — is what stops the row being retried forever. The same
   `maps_lookup_at IS NULL` predicate gates enrichment in `ml_prediction.py` and
   `train_bqml_model.py`; all three must agree or permanent misses get re-queried at cost.
3. **Gemini profiling** — `execute_gemini_enrichment` in `bq_utils.py` runs three SQL steps
   (identify recents → `AI.GENERATE` → MERGE) using the templates in `scripts/bq_scripts.py`. The
   result lands in `gemini_insights_structured` as raw JSON; the legacy text column
   `gemini_insights` is nulled on merge.
4. **Demographics** — `scripts/enrich_postcode_demographics.py` fills LSOA/MSOA/IMD from postcodes.io.
5. **Predict** — `app/services/ml_prediction.py` runs steps 2–4 just-in-time for whatever is missing,
   then `ML.PREDICT` into `predicted_user_rating` + `predicted_at`. Whether step 3 is "missing" is
   decided by `needs_gemini_profile` (`app/core/profile_freshness.py`) — no profile, or one older
   than `GEMINI_PROFILE_MAX_AGE_DAYS` (180) by `gemini_profiled_at`. The UI's "Estimated New Gemini
   Calls" calls the same predicate, so the estimate cannot drift from the spend; the training
   pre-flight calls it with `max_age_days=None`, because it is scheduled and must never refresh a
   profile it already has.
6. **Label** — the Streamlit UI writes `user_rating` (1–10), `in_scope`, and `rating_source`
   ("desk"/"visited") back via `bulk_update_reviews`, which is what
   `scripts/train_bqml_model.py` trains on. The loop closes here.

### Two separate Gemini surfaces — don't confuse them

- **The production profiler** is a BigQuery `AI.GENERATE` call. Its prompt is
  `_SYSTEM_INSTRUCTION_TEXT` in `scripts/bq_scripts.py` — the "Healthy Host & Explorer" persona and
  the 6 evaluation pillars. This is what the Streamlit app and the ML pipeline actually use.
- **The ADK agents** (`app/agent.py` root agent, `app/maps_agent/agent.py`) use
  `GoogleMapsGroundingTool` and are served by `app/fast_api_app.py`. **Streamlit never calls them.**
  They exist as the evaluated agent surface for `tests/eval/` and `adk eval`. The Docker image's
  `CMD` runs Streamlit, so the FastAPI app only runs locally or under test.

Both must stay on `gemini-3.8-flash` (or `gemini-3.1-pro`). Legacy model IDs are prohibited and
`tests/test_model_upgrades.py` asserts this across agents, SQL, and eval configs.

### The 6-pillar contract has one definition

`app/core/pillar_schema.py` is it. `PILLAR_FIELDS` — 14 `PillarField`s, each with a `column`, a
`bq_type`, the JSON `keys`, and an `is_feature` flag — generates the BigQuery columns, the
extraction SQL, the Python parser, the model feature list, and the conformance check. Adding or
renaming a field is a one-line change there. The paths are **nested**
(`"1_value_and_volume": {"rating": …}`), matching what the prompt emits; only `$.match_score` is
genuinely top-level.

`app/core/model_features.py` builds on it to define the model's input: `feature_select_list()` and
`feature_source_clause()` are called by *both* `build_training_select` (`train_bqml_model.py`) and
`build_prediction_input_select` (`ml_prediction.py`), so train/serve parity is structural rather
than a convention — `app/core/test_model_features.py::TestTrainServeParity` fails if the two
generated strings ever differ. Missing scores stay NULL; there is no `IFNULL(…, 0)`, because a
constant-zero feature is indistinguishable from an uninformative one.

**Changing the feature list changes the model's input schema**, so `ML.PREDICT` against a model
trained on the old one fails. Retrain in the same change.

The UI reads those columns straight out of BigQuery — `DISPLAY_COLUMNS` (`app/ui/st_app.py`) takes
its pillar half from `ALL_COLUMNS`, and nothing parses the JSON blob per row any more.
`gemini_insights_structured` is kept as the audit trail: it is the only way to re-derive a column
after a schema change.

The profiler's prompt concatenates row fields directly. **Every field must be `COALESCE`d** — a NULL
anywhere makes the whole prompt NULL and `AI.GENERATE` returns nothing, silently (D-16).

### Prioritization heuristic

`calculate_restaurant_priority` (`app/core/data_processing.py`) decides which restaurants are worth
spending Gemini calls on. It blends four weighted scores — proximity (exponential decay from an
anchor postcode, default SW16), staleness (unscored = 100, then tiered by `predicted_at` age), a
Google Maps quality prior, and scope confidence. Out-of-scope rows are forced to 0; rows that already
have a human `user_rating` are discounted to 10%. `st_app.py` exposes the weight vectors as strategy
presets. Distance falls back to outcode centroids from `app/core/london_outcodes.json` (310 UK
outcodes) when exact lat/lon is missing.

### UI shape

`app/ui/st_app.py` is one page: sidebar filters that push predicates down into the BigQuery `WHERE`
clause (`load_filtered_data_from_bq`), plus in-memory slicers (`filter_and_sort_restaurants`) over
the loaded DataFrame. Below the master grid, four action tabs operate on the current selection: Scope
Triage → Manual Rating → ML Predictions → Model Training. Selection state is deliberately reset
(`reset_selection_state`) after every write, because Streamlit returns positional row indices that go
stale when the underlying frame shrinks.

## Conventions

- **SQL is built by f-string interpolation**, not parameterized queries; string values go through
  `_sql_quote`. Python 3.11 forbids backslashes inside f-string expressions and several commits have
  fixed breakage from this — the `.venv` here is 3.13, but Docker and Cloud Build run **3.11**, so
  code that works locally can fail the build. Use `chr(39)`-style escapes as the existing code does.
- Unit tests live **beside** the code (`app/**/test_*.py`); integration, eval, and BQML tests live in
  `tests/`. `INTEGRATION_TEST=TRUE` disables live Cloud Trace export.
- Work is tracked under `conductor/tracks/<name>/plan.md` following the TDD workflow in
  `conductor/workflow.md`; `conductor/code_styleguides/` holds the Google Python style summary.
  Commits follow `type(scope): description`.
- `README.md` and `GEMINI.md` reference `agents-cli eval run …`; that CLI is not installed in `.venv`.
  Use `adk eval` (the `adk` binary is present) with `tests/eval/evalsets/restaurant_eval.evalset.json`.
