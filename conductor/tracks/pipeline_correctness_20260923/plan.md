# Plan: Pipeline Correctness & Simplification

Phases are ordered so that evidence precedes design, and the cost fixes that need neither ship
first. Phases 4 onward touch `fsa_master` and are gated on the Phase 0 snapshot and explicit
approval.

See `decision_log.md` for recorded measurements and decisions.

---

## Cross-cutting constraints

**No non-production environment exists.** One project, one dataset (`filipegracio_fsa_restaurants`),
one table, one Cloud Run service; even `recents` and `genairesults_temp` live in the production
dataset. The Phase 0 snapshot is therefore the only non-prod target available, which is why the
migration and backfill run against it first.

**`main` auto-deploys.** `cloudbuild.yaml` builds and deploys `restaurants-fsa` to Cloud Run; the
trigger config is server-side, so confirm with `gcloud builds triggers list` before relying on the
branch filter. Work happens on `track/pipeline-correctness`. Phase 1 merges to `main` as soon as it
is green; after that, merge only at phase boundaries that are green and independently deployable.

**CI gate.** `cloudbuild.yaml:11` runs `python -m pytest app/` in `python:3.11-slim` after
`pip install -r requirements.txt`. New script dependencies must reach `requirements.txt`, and
anything under `tests/` never runs in CI — so new tests belong under `app/`.

**Rollback.** Code: `git revert` the phase merge. Data: `CREATE OR REPLACE TABLE fsa_master AS
SELECT * FROM fsa_master_backup_20260923`. Schema: added columns are nullable and additive;
`DROP COLUMN` reverses them. Nothing is dropped from the live table before Phase 10.

**Approval gates.** No BigQuery write without `--dry-run` output shown first. `DROP COLUMN`,
snapshot deletion, and every Gemini re-profile sweep need explicit per-step go-ahead with a £
estimate attached.

---

## Phase 0: Safety Net and Reconnaissance (read-only + one snapshot)

*Gate: nothing from Phase 3 on is designed until these numbers exist.*

- [ ] Task: Establish the track workspace
    - [ ] Sub-task: Create branch `track/pipeline-correctness`.
    - [ ] Sub-task: Create `decision_log.md` in this track folder.
- [ ] Task: Snapshot the master table
    - [ ] Sub-task: Copy `fsa_master` → `fsa_master_backup_20260923`.
    - [ ] Sub-task: Verify row count and `COUNT(user_rating)` match the source.
    - [ ] Sub-task: Record both in `decision_log.md` as the restore reference.
- [ ] Task: Write and run `scripts/recon_pipeline_state.py` (read-only, `--dry-run` default)
    - [ ] Sub-task: Full top-level key census of `gemini_insights_structured` via `JSON_KEYS` +
          `UNNEST`, then one level deeper per pillar object. This settles which of the four key
          conventions is real **and** whether the shape is stable across rows (D14).
    - [ ] Sub-task: `COUNT(*) WHERE gemini_insights IS NOT NULL` — expected 0, confirming D1.
    - [ ] Sub-task: Distinct-value counts for each flat path the training/prediction SQL reads,
          confirming D2.
    - [ ] Sub-task: Does `'$.6_establishment_integrity_is_sit_down_restaurant'` ever resolve
          non-NULL? If not, count the `in_scope` rows assigned by `maps_types` alone (D13).
    - [ ] Sub-task: Sizing — total rows, profiled rows, `maps_rating = -1` count, rows with lat/lon,
          exact `COUNT(user_rating)`.
    - [ ] Sub-task: Capture real payloads to disk as test fixtures.
- [ ] Task: Reconcile the specification against the evidence
    - [ ] Sub-task: Record all numbers in `decision_log.md`.
    - [ ] Sub-task: Confirm or replace the provisional §4 column table.
- [ ] Task: Draft the cost ledger
    - [ ] Sub-task: Backfill is pure SQL over existing JSON — confirm ≈ free.
    - [ ] Sub-task: Price every candidate re-profile sweep as rows × `AI.GENERATE` unit cost, in £.

## Phase 1: Stop the Bleeding (no schema change, no recon dependency)

*Independent of Phase 0 — runs in parallel. Each task is individually shippable and revertible.*

- [x] Task: Stop the Gemini re-enrichment loop (D1) — 3147fb0
    - [x] Sub-task: `ml_prediction.py` selects and tests `gemini_insights_structured`, not
          `gemini_insights`, in both the targeted and the untargeted query branch.
    - [x] Sub-task: Regression test — a row with a populated profile but NULL `gemini_insights`
          (the production state) is excluded when `force_gemini` is false.
    - [x] Sub-task: Test that `force_gemini=True` still includes it.
    - [x] Sub-task: Test that the query selects the structured column.
- [ ] Task: Stop a Places miss from erasing coordinates (D5)
    - [ ] Sub-task: Remove `latitude`/`longitude` from the miss payload in `enrich_maps_data.py`.
    - [ ] Sub-task: Test that a miss leaves existing coordinates intact.
- [ ] Task: Stop the cron's full-table scan (D6)
    - [ ] Sub-task: Add an FHRSID-only loader to `bq_utils.py` returning a set.
    - [ ] Sub-task: Point `fetch_weekly.py` at it; drop `load_all_data_from_bq` if unused.
    - [ ] Sub-task: Update `app/cron/test_fetch_weekly.py`.
- [ ] Task: Bound ingest pagination (D7)
    - [ ] Sub-task: `max_pages` cap on `fetch_data_for_all_coordinates` with a conservative default.
    - [ ] Sub-task: Warn when the cap is hit.
    - [ ] Sub-task: Test that a mock API returning full pages forever terminates.
- [ ] Task: Isolate the temp tables (D10)
    - [ ] Sub-task: Per-run suffix and expiration on `recents` / `genairesults_temp`.
    - [ ] Sub-task: Wrap the `bulk_update_reviews` temp-table delete in `finally`.
- [ ] Task: Conductor — User Manual Verification 'Stop the Bleeding' (Protocol in workflow.md)
- [ ] Task: Merge to `main` and verify the Cloud Run deploy

## Phase 2: Baseline the Model Before Repairing It (R7)

*A held-out number taken now is the only thing that can prove the repair helped.*

- [ ] Task: Build the held-out evaluation harness
    - [ ] Sub-task: `scripts/evaluate_model.py`, with its scoring core under `app/` so CI covers it.
    - [ ] Sub-task: Split labelled rows, train on the majority, `ML.EVALUATE` on the remainder.
- [ ] Task: Record the two baseline numbers
    - [ ] Sub-task: MAE/RMSE for the current model, dead features and all.
    - [ ] Sub-task: The no-ML baseline — the same held-out rows ranked by `match_score` alone.
    - [ ] Sub-task: Write both to `decision_log.md` for verbatim reuse in Phase 9.

## Phase 3: Constrain the Profiler's Output Shape (D14)

*Design the columns against a guaranteed shape, not an observed one. Supersedes the original
spec §6 exclusion on prompt/model-params changes.*

- [ ] Task: Choose the mechanism and get approval
    - [ ] Sub-task: Present the tradeoff with costs. `tools: [{"googleSearch": {}}]` is enabled and
          grounded search is generally incompatible with constrained decoding, so `responseSchema`
          is not a drop-in addition.
    - [ ] Sub-task: Recommended — two-step: keep the grounded call, then a cheap *ungrounded*
          `AI.GENERATE` with `responseSchema` normalising the result into the canonical shape.
          Costs one extra call per profile but preserves the grounding the prompt depends on.
    - [ ] Sub-task: Alternative — drop `googleSearch` for direct constrained decoding. Cheaper, but
          removes grounding; the Phase 2 harness must quantify the regression before accepting it.
- [ ] Task: Define the canonical pillar schema once in code
    - [ ] Sub-task: Single source of truth for the response schema, the BigQuery columns, and the
          extraction paths.
- [ ] Task: Validate conformance
    - [ ] Sub-task: Run a small costed-and-approved sample; confirm 100% conformance.
    - [ ] Sub-task: Contract test running the real extraction against the Phase 0 fixtures — the
          current tests provably cannot catch this class of bug.
    - [ ] Sub-task: Record conformance in `decision_log.md`.

## Phase 4: Expand — Additive Schema

- [ ] Task: Write `scripts/migrate_pillar_columns.py`
    - [ ] Sub-task: Follow `scripts/migrate_to_in_scope_workflow.py` — `ADD COLUMN IF NOT EXISTS`,
          logged DDL, `--dry-run`, idempotent.
    - [ ] Sub-task: Columns from the Phase 3 canonical schema, plus `gemini_profiled_at`,
          `maps_lookup_at`, `maps_found`.
    - [ ] Sub-task: `pillar_geo_specificity` and `pillar_establishment_type` stay STRING — they are
          enums, and `CAST(... AS INT64)` is what silently zeroes pillar 4 today.
- [ ] Task: Review and execute
    - [ ] Sub-task: Present dry-run output for approval.
    - [ ] Sub-task: Execute against the snapshot table first; validate; then prod.
    - [ ] Sub-task: Verify nothing changed — row count and per-column checksum before/after.
    - [ ] Sub-task: **Ordering** — `ALTER TABLE` lands *before* `MASTER_BQ_SCHEMA` is updated. That
          constant is the load schema for the weekly cron's `append_to_bigquery`; if it lists a
          column the live table lacks, the scheduled ingest fails. Nothing reconciles the two.

## Phase 5: Backfill and Validate

*Legacy rows hold the old shape, new rows the canonical one. The backfill bridges both.*

- [ ] Task: Write `scripts/backfill_pillar_columns.py`
    - [ ] Sub-task: Populate typed columns from `gemini_insights_structured`, `COALESCE`-ing across
          every alias the Phase 0 census actually observed.
    - [ ] Sub-task: `--dry-run` reporting affected rows and per-column non-null tallies.
- [ ] Task: Decide the legacy-row strategy
    - [ ] Sub-task: Present, with £ figures, whether legacy rows are re-profiled into the canonical
          shape or left to alias-extraction. Await approval before running either.
- [ ] Task: Backfill the timestamps
    - [ ] Sub-task: `gemini_profiled_at` ← `CURRENT_TIMESTAMP()` for rows that already have a
          profile, following `scripts/migrate_predicted_at.py`. Existing profiles count as fresh, so
          the first stale sweep is a budget-capped operation rather than a full-table re-profile.
- [ ] Task: Backfill the Maps sentinel (D4, R3)
    - [ ] Sub-task: `maps_rating = -1` rows → `maps_found = FALSE`, `maps_lookup_at` set to a past
          timestamp, `maps_rating`/`maps_reviews` nulled.
    - [ ] Sub-task: Real ratings → `maps_found = TRUE` and `maps_lookup_at` set.
    - [ ] Sub-task: The timestamp must take over the do-not-retry role `-1` plays in
          `enrich_maps_data.py`, or every run re-queries Places for permanent misses.
    - [ ] Sub-task: Verify `COUNT(*) WHERE maps_rating = -1` is zero afterwards.
- [ ] Task: Re-derive `in_scope` if recon showed the triage path never resolved (D13)
    - [ ] Sub-task: Derive from the now-typed `pillar_is_sit_down` for affected rows.
    - [ ] Sub-task: Dry-run and diff count first — `in_scope` governs what gets profiled at all.
- [ ] Task: Validate the backfill
    - [ ] Sub-task: Every pillar column has more than one distinct value.
    - [ ] Sub-task: Spot-check a sample against the raw JSON.
    - [ ] Sub-task: Record per-column non-null coverage in `decision_log.md`.
    - [ ] Sub-task: Run on the snapshot copy first, compare, then prod.

## Phase 6: Dual-Write

*Writers populate both representations so the deployed app keeps working mid-migration.*

- [ ] Task: Populate the new columns going forward
    - [ ] Sub-task: `SCRIPT_MERGE_INSIGHTS` writes the typed columns and
          `gemini_profiled_at = CURRENT_TIMESTAMP()`, keeping `gemini_insights_structured` as the
          raw audit trail.
    - [ ] Sub-task: The Places merge writes `maps_found` / `maps_lookup_at` and stops writing `-1`.
    - [ ] Sub-task: Switch the "needs Maps" guard from `maps_rating IS NULL` to
          `maps_lookup_at IS NULL`, preserving do-not-retry.
- [ ] Task: Single source of truth for model features (D3)
    - [ ] Sub-task: Build both the training `SELECT` and the `ML.PREDICT` subquery from the Phase 3
          canonical schema.
    - [ ] Sub-task: Test that fails if the two feature sets diverge.

## Phase 7: Switch Readers, Stale-Aware Refresh (R1)

- [ ] Task: Read typed columns instead of parsing JSON
    - [ ] Sub-task: Reduce `parse_insight_row` to reading columns; drop the V1 text branch.
    - [ ] Sub-task: The per-row `json.loads` in `enhance_dataframe_with_insights` disappears.
    - [ ] Sub-task: Update `DISPLAY_COLUMNS`.
- [ ] Task: Stale-aware refresh
    - [ ] Sub-task: Threshold against `gemini_profiled_at` as a named constant.
    - [ ] Sub-task: The UI's "Estimated New Gemini Calls" uses the **same predicate** as the
          executor, so estimate and behaviour cannot drift apart again — that divergence is D1.
    - [ ] Sub-task: The first stale sweep is budget-capped, with its £ cost stated before it runs.
    - [ ] Sub-task: Tests for fresh / stale / never-profiled / forced.
- [ ] Task: Conductor — User Manual Verification 'Rewire' (Protocol in workflow.md)

## Phase 8: Honest Missing Data and Free Coordinates (D4, R4)

- [ ] Task: Missing location stops scoring as perfect
    - [ ] Sub-task: `extract_outcode("")` stops returning `"SW16"`; return an explicit unknown.
    - [ ] Sub-task: Unknown-location rows get a low-or-neutral proximity score, not the maximum.
    - [ ] Sub-task: Update `app/core/test_scoring_priority.py`, which asserts the current default.
    - [ ] Sub-task: Confirm on real data that the top of the queue moves as expected.
- [ ] Task: Keep FSA coordinates at ingest
    - [ ] Sub-task: Flatten the API's nested `Geocode` into `latitude`/`longitude` in
          `process_and_update_master_data`.
    - [ ] Sub-task: Add them to `ORIGINAL_COLUMNS_TO_KEEP`, which currently drops `Geocode`.
    - [ ] Sub-task: Remove the dead `Geocode.Latitude` handling from `write_to_bigquery`.
    - [ ] Sub-task: Test that ingest stores coordinates (the miss-path erasure is already fixed in
          Phase 1).
- [ ] Task: Purge sentinel awareness from readers
    - [ ] Sub-task: UI metric, "Has Google Maps Rating" filter, and sorting read `maps_found`.
    - [ ] Sub-task: Confirm no feature column can receive `-1`.

## Phase 9: Retrain and Deliver the Verdict

- [ ] Task: Retrain on corrected features
    - [ ] Sub-task: `--dry-run` first to validate the generated SQL.
    - [ ] Sub-task: Train and compare against the Phase 0 baseline.
- [ ] Task: Re-run the Phase 2 harness unchanged
    - [ ] Sub-task: Report repaired model vs baseline model vs `match_score`-only.
    - [ ] Sub-task: Write the keep-or-retire recommendation for BQML into `decision_log.md`.
          Retiring it would be a separate track; this phase produces evidence only.
- [ ] Task: Invalidate stale predictions
    - [ ] Sub-task: Clear `predicted_user_rating` / `predicted_at` for rows scored by the old model.
    - [ ] Sub-task: Confirm the queue repopulates as expected.

## Phase 10: Contract — Remove Legacy Surfaces (R5)

*Stop writing first; drop columns last.*

- [ ] Task: Retire `manual_review`
    - [ ] Sub-task: **Replace**, don't delete, the default filter in `execute_gemini_enrichment`
          with an `in_scope`-based predicate — a behaviour change, not a deletion.
    - [ ] Sub-task: Remove from ingest, `bulk_update_reviews`, `DISPLAY_COLUMNS`, filter signatures.
    - [ ] Sub-task: Update the affected tests.
- [ ] Task: Retire `gemini_insights` (V1 text)
    - [ ] Sub-task: Remove the `gemini_insights_status` filter branch and stop nulling the column.
    - [ ] Sub-task: Remove from `ORIGINAL_COLUMNS_TO_KEEP`.
- [ ] Task: Remove the non-functional sidebar path input
    - [ ] Sub-task: Also closes the SQL-injection path recorded in the teamwork handoff note; the
          broader f-string SQL interpolation stays out of scope.
- [ ] Task: Remove `app/maps_agent/`
    - [ ] Sub-task: Delete the package and its test.
    - [ ] Sub-task: Remove its assertions from `tests/test_model_upgrades.py` and eval config refs.
    - [ ] Sub-task: Confirm `app/agent.py` still loads and the ADK server still starts.
- [ ] Task: Drop the retired columns (destructive — separate approval)
    - [ ] Sub-task: Confirm by grep that nothing reads them.
    - [ ] Sub-task: Confirm the Phase 0 snapshot still exists.
    - [ ] Sub-task: `ALTER TABLE ... DROP COLUMN` only on explicit go-ahead.

## Phase 11: Errors, Performance, Hygiene

*Deliberately last — cross-cutting churn that would otherwise have blocked the cost fixes.*

- [ ] Task: Make failures visible (D9)
    - [ ] Sub-task: BigQuery helpers raise, or return an error-carrying result, instead of
          `[]`/`False`.
    - [ ] Sub-task: Surface the message in the UI so an auth failure stops reading as "No data
          found matching criteria."
    - [ ] Sub-task: Adjust the tests that assert the swallowing behaviour.
- [ ] Task: Vectorise priority scoring (D8)
    - [ ] Sub-task: Replace `.iterrows()` with vectorised pandas/numpy.
    - [ ] Sub-task: Hoist the 310-key `sorted()` out of the per-row path into a precomputed lookup.
    - [ ] Sub-task: Cache so a Streamlit rerun does not recompute it twice.
    - [ ] Sub-task: Assert identical output on a fixture before/after.
- [ ] Task: Make the test suite runnable (D11)
    - [ ] Sub-task: Mark live tests `integration`; register markers in `pyproject.toml`.
    - [ ] Sub-task: Default a bare `pytest` to offline tests only.
    - [ ] Sub-task: Document how to opt into the live suite.
- [ ] Task: Unify dependencies (D12)
    - [ ] Sub-task: Make `pyproject.toml`/`uv.lock` the single source; generate `requirements.txt`.
    - [ ] Sub-task: Pinning local Python to 3.11 is **optional and deferred** — it means rebuilding
          a `.venv` that currently works on 3.13, for parity benefit only.
- [ ] Task: Simplify the filter API
    - [ ] Sub-task: Collapse the ~3 accepted string aliases per filter to a single set of constants.
    - [ ] Sub-task: Drop the legacy `rating_filter` / `pred_filter` kwargs.
- [ ] Task: Conductor — User Manual Verification 'Performance & Hygiene' (Protocol in workflow.md)

## Phase 12: Close Out

- [ ] Task: Full regression — `pytest app/`, then the live suite deliberately
- [ ] Task: Update `CLAUDE.md` — the pillar-mismatch and legacy-surface notes become obsolete
- [ ] Task: Reconcile `README.md` / `GEMINI.md` (both still document the uninstalled `agents-cli`)
- [ ] Task: Deploy and verify on Cloud Run
- [ ] Task: Drop the Phase 0 snapshot once the new pipeline has run clean for a full cycle
