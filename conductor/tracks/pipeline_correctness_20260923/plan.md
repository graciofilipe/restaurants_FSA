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

- [x] Task: Establish the track workspace — d19d204
    - [x] Sub-task: Create branch `track/pipeline-correctness`.
    - [x] Sub-task: Create `decision_log.md` in this track folder.
- [x] Task: Snapshot the master table — 0a00240
    - [x] Sub-task: Copy `fsa_master` → `fsa_master_backup_20260923`.
    - [x] Sub-task: Verify row count and `COUNT(user_rating)` match the source. **11,268 / 411,
          matching exactly.**
    - [x] Sub-task: Record both in `decision_log.md` as the restore reference.
    - *Deviation:* the copy uses `CREATE TABLE IF NOT EXISTS`, not `CREATE OR REPLACE`. Re-running
      the script after a later phase has modified `fsa_master` would otherwise overwrite the restore
      point with the very state it exists to protect.
- [x] Task: Write and run `scripts/recon_pipeline_state.py` (read-only, `--dry-run` default) — 0a00240
    - [x] Sub-task: Full top-level key census of `gemini_insights_structured` via `JSON_KEYS` +
          `UNNEST`, then one level deeper per pillar object. This settles which of the four key
          conventions is real **and** whether the shape is stable across rows (D14).
          **Convention A, stable: 2,766/2,767 rows share one identical key set.**
    - [x] Sub-task: `COUNT(*) WHERE gemini_insights IS NOT NULL` — expected 0, confirming D1.
          **It is 1,116, not 0** — see D-08; the defect is worse than the plan assumed, not milder.
    - [x] Sub-task: Distinct-value counts for each flat path the training/prediction SQL reads,
          confirming D2. **All five resolve on zero rows; only `$.match_score` resolves.**
    - [x] Sub-task: Does `'$.6_establishment_integrity_is_sit_down_restaurant'` ever resolve
          non-NULL? If not, count the `in_scope` rows assigned by `maps_types` alone (D13).
          **Never resolves; 1,476 of 2,766 profiled rows contradict their own profile.**
    - [x] Sub-task: Sizing — total rows, profiled rows, `maps_rating = -1` count, rows with lat/lon,
          exact `COUNT(user_rating)`. **11,268 / 2,767 / 243 / 2,291 / 411.**
    - [x] Sub-task: Capture real payloads to disk as test fixtures.
          **8 in `tests/fixtures/gemini_profiles/`.**
    - *Deviation:* implemented as an opt-in `--execute` rather than an opt-out `--dry-run`. Same
      behaviour, but it reads as a default instead of a flag you must remember. `JSON_KEYS(json, 2)`
      returns both levels at once, so the depth-2 census is one query rather than seven.
- [x] Task: Reconcile the specification against the evidence — 0a00240
    - [x] Sub-task: Record all numbers in `decision_log.md`.
    - [x] Sub-task: Confirm or replace the provisional §4 column table. **Confirmed unchanged** —
          all 17 source paths resolve on all 2,766 rows; the PROVISIONAL banner is removed.
- [x] Task: Draft the cost ledger — 0a00240
    - [x] Sub-task: Backfill is pure SQL over existing JSON — confirm ≈ free. **Confirmed, < £0.01.**
    - [x] Sub-task: Price every candidate re-profile sweep as rows × `AI.GENERATE` unit cost, in £.
          **The legacy re-profile is withdrawn entirely** (the existing JSON already holds every
          field the typed columns need). The only sweep left needing approval is Phase 7's, over the
          1,116 V1-only rows Phase 1's D1 fix newly exposes.

**Exit met.** Observed shape and its stability documented; spec §4 confirmed; snapshot verified at
11,268 / 411; cost ledger in `decision_log.md`. Two findings change later phases: Phase 3 is
downscoped to a conformance check, and Phase 5 gains a D13 `in_scope` re-derivation. Both recorded
in D-08.

## Phase 1: Stop the Bleeding (no schema change, no recon dependency) [checkpoint: 59439e3]

*Independent of Phase 0 — runs in parallel. Each task is individually shippable and revertible.*

- [x] Task: Stop the Gemini re-enrichment loop (D1) — 3147fb0
    - [x] Sub-task: `ml_prediction.py` selects and tests `gemini_insights_structured`, not
          `gemini_insights`, in both the targeted and the untargeted query branch.
    - [x] Sub-task: Regression test — a row with a populated profile but NULL `gemini_insights`
          (the production state) is excluded when `force_gemini` is false.
    - [x] Sub-task: Test that `force_gemini=True` still includes it.
    - [x] Sub-task: Test that the query selects the structured column.
- [x] Task: Stop a Places miss from erasing coordinates (D5) — 3e5f0c5
    - [x] Sub-task: Guard `latitude`/`longitude` in the `enrich_maps_data.py` MERGE with
          `IFNULL(S.x, T.x)`. *Deviation:* the plan said to drop the keys from the miss payload, but
          that payload feeds a positional STRUCT whose field list must stay intact. Guarding in the
          MERGE is correct and also covers a Places **hit** that returns no location block — a case
          the plan had not identified.
    - [x] Sub-task: Test that a miss leaves existing coordinates intact.
    - [x] Sub-task: First tests for this script; `scripts/` added to the Cloud Build test command.
- [x] Task: Stop the cron's full-table scan (D6) — 73e039c
    - [x] Sub-task: Add an FHRSID-only loader to `bq_utils.py` returning a set. *Deviation:* it
          raises `BigQueryExecutionError` instead of returning an empty set. The module-wide swallow
          is D9/Phase 11 work, but here "read failed" and "table is empty" call for opposite actions
          — an empty set would re-append the whole fetch — and `fetch_weekly.py:89` already aborts
          the sync on an exception.
    - [x] Sub-task: Point `fetch_weekly.py` at it; drop `load_all_data_from_bq` if unused. Both it
          and `load_master_data` (its only wrapper) were callerless after the switch and are gone,
          with their 10 tests. `load_master_data` also injected the `manual_review` defaults that
          Phase 10 removes anyway.
    - [x] Sub-task: `process_and_update_master_data` accepts bare FHRSIDs alongside full rows.
          *Not in the plan, but required:* its `isinstance(est, dict)` guard would have silently
          dropped a set of strings, making every restaurant look new and duplicating the table.
          The two copies of the ID-normalisation logic collapse into `normalize_fhrsid`.
    - [x] Sub-task: Update `app/cron/test_fetch_weekly.py` — loads IDs only, and aborts without
          appending when the load fails.
- [x] Task: Bound ingest pagination (D7) — d3cdddb
    - [x] Sub-task: `max_pages` cap on `fetch_data_for_all_coordinates` with a conservative default.
          50, and deliberately loose: it is a runaway guard, not an expected limit. Production
          configs pass `max_results=5000`, so real runs break on page 1; a cap tight enough to
          truncate a legitimate ingest would be worse than the bug. Applied **per coordinate** — a
          shared counter would let one exhausted coordinate shorten the next one's fetch.
    - [x] Sub-task: Warn when the cap is hit, via `for`/`else` so it fires only when the loop was
          never broken out of — the case where truncation and a misbehaving API are indistinguishable.
    - [x] Sub-task: Test that a mock API returning full pages forever terminates.
- [x] Task: Isolate the temp tables (D10) — 96196fc
    - [x] Sub-task: Per-run suffix and expiration on `recents` / `genairesults_temp`, plus a
          `finally` that drops both. All three layers are deliberate: expiry alone leaves a day of
          clutter per run, and a `finally` alone does not survive the instance being killed mid-run,
          which is how the current leak happens.
    - [x] Sub-task: Wrap the `bulk_update_reviews` temp-table delete in `finally`.
    - [x] *Follow-up for the checkpoint:* the existing production `recents` and `genairesults_temp`
          tables are now orphaned. Dropping them is destructive and needs explicit go-ahead.
- [x] Task: Conductor — User Manual Verification 'Stop the Bleeding' (Protocol in workflow.md) — 59439e3
    - [x] Sub-task: `pytest app/ scripts/` → 106 passed; all sources parsed against the 3.11 grammar
          CI builds on. Coverage was **not** measured — `pytest-cov` is in neither `.venv` nor
          `requirements.txt`, so `workflow.md`'s >80% gate could not be evaluated. Folded into
          D11/D12 in Phase 11.
    - [x] Sub-task: Added `scripts/test_bq_scripts.py` — the protocol requires a test file per
          changed code file, and `bq_scripts.py` had none despite holding SQL run against prod.
    - [x] Sub-task: Verified the deploy trigger with `gcloud` before relying on it — see D-07. It is
          real, and region-scoped to `europe-west2`.
- [x] Task: Merge to `main` and verify the Cloud Run deploy — 89227f3
    - [x] Sub-task: Merged `--no-ff`; `pytest app/ scripts/` re-run on the merge commit → 106 passed.
    - [x] Sub-task: Cloud Build `6479c01b` SUCCESS on all four steps
          (`run-tests`, `build-image`, `push-image`, `deploy-cloud-run`).
    - [x] Sub-task: Deploy verified by digest, not by the build status alone — live revision
          `restaurants-fsa-00218-cs2` runs
          `restaurants-fsa@sha256:e4da6388…`, which Artifact Registry tags `89227f39…`, the merge
          commit. Service returns HTTP 302 (Streamlit's normal redirect).

## Phase 2: Baseline the Model Before Repairing It (R7)

*A held-out number taken now is the only thing that can prove the repair helped.*

- [x] Task: Build the held-out evaluation harness — 9c2b0e0
    - [x] Sub-task: `scripts/evaluate_model.py`, with its scoring core under `app/` so CI covers it.
          *Deviation:* it lives wholly in `scripts/`. The constraint existed because CI ran only
          `pytest app/`; Phase 1 changed `cloudbuild.yaml` to `pytest app/ scripts/`, so `scripts/`
          is now covered and the split would have bought nothing.
    - [x] Sub-task: Split labelled rows, train on the majority, `ML.EVALUATE` on the remainder.
          **292 train / 77 holdout**, split on `FARM_FINGERPRINT(fhrsid) MOD 5` so Phase 9 gets the
          same rows. Mean rating 2.613 vs 2.299 — close enough that the comparison is not measuring
          the split.
    - [x] Sub-task: Extract `build_training_select` from `train_bqml_model.py` so the harness
          baselines the production features rather than a third hand-copy. *Not in the plan;*
          required, because a hand-copied feature list would have made the Phase 9 delta measure
          harness drift. The production training query was dry-run afterwards to confirm it is
          unchanged.
- [x] Task: Record the two baseline numbers — 9c2b0e0
    - [x] Sub-task: MAE/RMSE for the current model, dead features and all. **0.847 / 1.283.**
    - [x] Sub-task: The no-ML baseline — the same held-out rows ranked by `match_score` alone.
          **0.697 / 1.095, Spearman 0.603.** Fitted as a one-feature `LINEAR_REG`, since
          `match_score` is 0–100 and `user_rating` is 1–10 and a raw MAE would have measured the
          scale gap.
    - [x] Sub-task: A third number not in the plan — the **training-mean floor, MAE 1.566**. Without
          it, "the tree scores 0.847" has no scale.
    - [x] Sub-task: Write both to `decision_log.md` for verbatim reuse in Phase 9. **D-11.**

**Exit met, with a result that reframes Phase 9.** The one-feature baseline beats the twenty-feature
boosted tree on every metric — MAE, RMSE, R² and rank correlation. Both beat the mean, so
`match_score` carries real signal and the tree destroys part of it. This is what five constant-zero
features plus high-cardinality categoricals over 292 rows should be expected to produce. The
keep-or-retire recommendation stays in Phase 9; Phase 2's job was to make it answerable.

## Phase 3: Constrain the Profiler's Output Shape (D14)

> **Downscoped by the Phase 0 recon — see D-08.** The drift that justified this phase came from the
> ADK recordings, not the profiler. The profiler is at 2,766/2,766 conformance on every required
> key, so neither the two-step normaliser nor dropping `googleSearch` is warranted: both would spend
> real money and real grounding quality to fix something that is not currently broken. What remains
> is making the shape *checked* rather than merely observed, so a future model revision surfaces as
> a failure instead of five more silent zeros.

*Define the columns against a schema that exists in one place, and detect drift rather than absorb
it. The original spec §6 exclusion on prompt/model-params changes is no longer being superseded —
no prompt or model-params change is now planned.*

- [x] Task: Choose the mechanism and get approval — decided on the recon evidence, D-08
    - [x] Sub-task: Two-step ungrounded normaliser — **rejected**: one extra `AI.GENERATE` per
          profile to normalise output that already conforms.
    - [x] Sub-task: Drop `googleSearch` for direct constrained decoding — **rejected**: trades the
          grounding the "Culinary Anthropologist" prompt depends on for the same non-benefit.
    - [x] Sub-task: Detect-not-prevent — **chosen**. Zero marginal cost, and it is the part that was
          actually missing: nothing in the repo would have told us the paths were dead.
- [x] Task: Define the canonical pillar schema once in code — 900fde9
    - [x] Sub-task: Single source of truth for the BigQuery columns, the extraction paths, and the
          conformance check. Spec §4 is the confirmed content. **`app/core/pillar_schema.py`: 14
          `PillarField`s (column, BigQuery type, key path, `is_feature`) plus `NON_JSON_COLUMNS`.
          The extraction SQL, the Python parser, the feature list, and the conformance query are all
          generated from that tuple, so Phase 4's DDL and Phase 6's dual-write cannot disagree with
          it.**
    - *Deviation:* the schema also carries `is_feature`, which the plan had not asked for. The eight
      feature columns are what Phase 6's D3 task needs a single definition of, and putting the flag
      beside the path is what makes "training and prediction agree" a test rather than a convention.
      It also settles a real question the plan left open: the four free-text fields are **not**
      features. Unbounded model prose as a training input is memorisation, not signal.
- [x] Task: Validate conformance — 900fde9, 061a197
    - [x] Sub-task: Contract test running the real extraction against the Phase 0 fixtures in
          `tests/fixtures/gemini_profiles/` — the current tests provably cannot catch this class of
          bug, which is how D2 survived. Include the one unparseable payload as a negative case.
          **`app/core/test_pillar_schema.py`, 46 tests.** Two fixtures were added beyond Phase 0's
          eight: the unparseable payload (1855447) and a markdown-fenced one (1040595), because the
          eight captured rows were all the plain shape and a contract test that only sees the happy
          case is the same blind spot again.
    - [x] Sub-task: Conformance check at merge time: a profile missing a required path is counted
          and logged, not silently merged as zeros. **`log_insight_conformance` in `bq_utils.py`,
          called against the scratch table immediately before `SCRIPT_MERGE_INSIGHTS`.**
    - [x] Sub-task: Record the check's first production reading in `decision_log.md`. **2,766 of
          2,767 conform on all 14 paths; the 1 failure is the unparseable row; no partial drift.
          See D-12.**
    - *Deviation:* the check is **advisory** — it logs and returns, and cannot fail an enrichment
      run. The plan said "counted and logged", which this satisfies, but the choice deserves stating:
      at 2,766/2,767, blocking the merge would throw away a run of good profiles over one leaked
      reasoning trace. Reasoning in D-12.
    - *Deviation:* `sql_conformance_check` gained a `where` parameter. The default must stay
      unfiltered for the scratch table, where a NULL `AI.GENERATE` result is a failed profile, but
      auditing `fsa_master` without a predicate drops all 8,501 never-profiled rows into the
      unparseable count. Both behaviours are pinned by tests.

**Phase 3 checkpoint:** `pytest app/ scripts/` green at 184 tests; all four new/edited files parse
under `ast.parse(feature_version=(3,11))` for Cloud Build parity. No prompt change, no model-params
change, no BigQuery write — the only production query run was the read-only conformance reading
(4.5 MiB, £0.00003). Phase 4 can now generate its DDL from `PILLAR_FIELDS` + `NON_JSON_COLUMNS`
rather than from a hand-copied column list.

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

> **Simplified by the Phase 0 recon — see D-08.** There is no old shape and no new shape: every one
> of the 2,767 profiled rows already carries all 17 fields under the same nested paths. The backfill
> is a straight extraction with no alias handling, and no legacy row needs re-profiling.

- [ ] Task: Write `scripts/backfill_pillar_columns.py`
    - [ ] Sub-task: Populate the typed columns from `gemini_insights_structured` using the spec §4
          nested paths directly. **No alias `COALESCE`** — the census found no alternate spellings
          of any path the schema reads.
    - [ ] Sub-task: Handle the 1 unparseable payload and the 22 markdown-fenced ones; the
          `REGEXP_EXTRACT` unwrap covers the fences, the unparseable row must land as NULL rather
          than failing the statement.
    - [ ] Sub-task: `--dry-run` reporting affected rows and per-column non-null tallies.
          Expect 2,766 non-null for every pillar column.
- [x] Task: Decide the legacy-row strategy — **withdrawn**, D-08
    - [x] Sub-task: Re-profiling legacy rows would cost a full 2,767-row Gemini sweep to produce
          fields the stored JSON already contains. Not run, not offered.
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
- [ ] Task: Re-derive `in_scope` — recon confirmed the triage path never resolved (D13)
    - [ ] Sub-task: Derive from the now-typed `pillar_is_sit_down` for the affected rows. The census
          measured **1,476 disagreements** out of 2,766 profiled rows: 1,458 marked in-scope that
          the profile calls not-sit-down, and 18 the other way.
    - [ ] Sub-task: Dry-run and diff count first — `in_scope` governs what gets profiled at all, and
          this moves ~1,458 rows *out* of scope, which is a large change to future Gemini spend.
    - [ ] Sub-task: Leave the 8,501 never-profiled rows alone; there is nothing to derive from.
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

- [ ] Task: Make `--dry-run` actually dry (D15, found in Phase 2 — see D-09)
    - [ ] Sub-task: Move the JIT pre-flight block at `train_bqml_model.py:26-60` inside the
          non-dry-run branch. It currently runs first and can issue grounded `AI.GENERATE` calls for
          the 7 labelled rows with no profile.
    - [ ] Sub-task: Test that a dry run triggers no enrichment call.
    - [ ] Sub-task: `CLAUDE.md` documents the flag as "validate BQML training SQL without spending";
          that becomes true rather than needing a correction.
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
