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

- [x] Task: Write `scripts/migrate_pillar_columns.py` — 06212ed
    - [x] Sub-task: Follow `scripts/migrate_to_in_scope_workflow.py` — `ADD COLUMN IF NOT EXISTS`,
          logged DDL, `--dry-run`, idempotent.
    - [x] Sub-task: Columns from the Phase 3 canonical schema, plus `gemini_profiled_at`,
          `maps_lookup_at`, `maps_found`. **17, generated from `PILLAR_FIELDS` + `NON_JSON_COLUMNS`,
          not retyped.**
    - [x] Sub-task: `pillar_geo_specificity` and `pillar_establishment_type` stay STRING — they are
          enums, and `CAST(... AS INT64)` is what silently zeroes pillar 4 today. **Pinned by test.**
    - *Deviation:* three departures from the script the plan says to follow, all recorded in D-13.
      Dry run is the **default** here (`--execute` opts in); a failed `ALTER TABLE` **raises**
      rather than logging a notice; and the dry run **submits** each statement to BigQuery with
      `dry_run=True` instead of printing it. The third caught a real error immediately —
      `COUNT(*) AS rows` is a syntax error, `ROWS` being reserved for window frames.
- [x] Task: Review and execute — 06212ed, 45d77aa
    - [x] Sub-task: Present dry-run output for approval. **All 17 statements validated against the
          live table; approved 2026-09-23, "snapshot first, then prod".**
    - [x] Sub-task: Execute against the snapshot table first; validate; then prod. **Snapshot:
          44 columns, correct types, all nullable, fingerprint unchanged. Then production: same.**
    - [x] Sub-task: Verify nothing changed — row count and per-column checksum before/after.
          **`rows=11268 labels=411 hash=7075448033881697774` before, after, and after a second
          idempotent run. The snapshot carried the identical hash pre-migration, so the restore
          point is byte-for-byte production, not merely the same shape.**
    - [x] Sub-task: **Ordering** — `ALTER TABLE` lands *before* `MASTER_BQ_SCHEMA` is updated. That
          constant is the load schema for the weekly cron's `append_to_bigquery`; if it lists a
          column the live table lacks, the scheduled ingest fails. Nothing reconciles the two.
          **Held: the DDL ran first, the constant was updated after, and Phase 4 stayed unmerged
          until both were done. Verified live — 44 declared, 44 present, none missing.**
    - *Deviation:* updating `MASTER_BQ_SCHEMA` broke the migration's own fingerprint, because
      `PRE_EXISTING_COLUMNS` was derived from it. Caught by the existing tests. It now subtracts
      `NEW_COLUMNS`, with a test that the subtraction cannot empty the list. See D-13.

**Phase 4 checkpoint:** `pytest app/ scripts/` green at 204; 3.11 parity checked. Both tables at 44
columns; data fingerprint identical across every run. Nothing reads the new columns yet, so the
deployed app is unaffected. Cost: £0 — `ADD COLUMN` is metadata-only.

## Phase 5: Backfill and Validate

> **Simplified by the Phase 0 recon — see D-08.** There is no old shape and no new shape: every one
> of the 2,767 profiled rows already carries all 17 fields under the same nested paths. The backfill
> is a straight extraction with no alias handling, and no legacy row needs re-profiling.

- [x] Task: Write `scripts/backfill_pillar_columns.py` — c609e34
    - [x] Sub-task: Populate the typed columns from `gemini_insights_structured` using the spec §4
          nested paths directly. **No alias `COALESCE`** — the census found no alternate spellings
          of any path the schema reads. Generated from `PILLAR_FIELDS` via `sql_extract`, so the
          paths are the same objects the parser and the conformance check use.
    - [x] Sub-task: Handle the 1 unparseable payload and the 22 markdown-fenced ones; the
          `REGEXP_EXTRACT` unwrap covers the fences, the unparseable row must land as NULL rather
          than failing the statement. Confirmed: 2,766 of 2,767 filled on every column.
    - [x] Sub-task: `--dry-run` reporting affected rows and per-column non-null tallies.
          Expect 2,766 non-null for every pillar column. *Deviation:* the tallies moved to a
          separate `--validate` mode. The dry run previews **affected rows per statement**, which
          is the number that governs approval; coverage is only meaningful after the write.
    - [x] Sub-task: *Not in the plan:* `--dry-run` submits every `UPDATE` to BigQuery with
          `dry_run=True` rather than printing it, as Phase 4 established. Each count query carries
          the same predicate as its `UPDATE`, so the estimate cannot drift from the behaviour —
          that divergence is the shape of D1.
- [x] Task: Decide the legacy-row strategy — **withdrawn**, D-08
    - [x] Sub-task: Re-profiling legacy rows would cost a full 2,767-row Gemini sweep to produce
          fields the stored JSON already contains. Not run, not offered.
- [x] Task: Backfill the timestamps — c609e34
    - [x] Sub-task: `gemini_profiled_at` ← `CURRENT_TIMESTAMP()` for rows that already have a
          profile, following `scripts/migrate_predicted_at.py`. Existing profiles count as fresh, so
          the first stale sweep is a budget-capped operation rather than a full-table re-profile.
          Guarded on `gemini_profiled_at IS NULL` so a re-run cannot slide the staleness clock.
- [x] Task: Backfill the Maps sentinel (D4, R3) — c609e34, 4cf568b
    - [x] Sub-task: `maps_rating = -1` rows → `maps_found = FALSE`, `maps_lookup_at` set to a past
          timestamp, `maps_rating`/`maps_reviews` nulled. 243 rows.
    - [x] Sub-task: Real ratings → `maps_found = TRUE` and `maps_lookup_at` set. 2,263 rows. Ordered
          before the miss statement, which erases the sentinel the hit statement selects against.
    - [x] Sub-task: The timestamp must take over the do-not-retry role `-1` plays in
          `enrich_maps_data.py`, or every run re-queries Places for permanent misses.
          *Deviation:* **three** sites made that decision, not one — `enrich_maps_data.py:26`,
          `ml_prediction.py:41` and `train_bqml_model.py:88` all read `maps_rating IS NULL` as
          "needs Maps". The plan assigned this to Phase 6; it was pulled forward and shipped
          immediately after the production backfill, on your call, because the window between the
          two is a window in which 243 rows are eligible for a paid Places re-query.
    - [x] Sub-task: *Not in the plan:* the **writer** changes too. A Places miss now records
          `maps_found = FALSE` and a lookup timestamp instead of a fresh `-1`. Without it the guard
          move is a one-way door — the next miss would put itself straight back in the queue.
    - [x] Sub-task: Verify `COUNT(*) WHERE maps_rating = -1` is zero afterwards. It is.
- [x] Task: Re-derive `in_scope` — recon confirmed the triage path never resolved (D13) — c609e34
    - [x] Sub-task: Derive from the now-typed `pillar_is_sit_down` for the affected rows. The census
          measured **1,476 disagreements** out of 2,766 profiled rows: 1,458 marked in-scope that
          the profile calls not-sit-down, and 18 the other way. Re-measured against the backfilled
          column: **1,479** (1,458 / 18 / 3 filling a NULL).
    - [x] Sub-task: Dry-run and diff count first — `in_scope` governs what gets profiled at all, and
          this moves ~1,458 rows *out* of scope, which is a large change to future Gemini spend.
          **This is where the plan was wrong.** 216 of the 1,479 disagreements carry a human
          `user_rating` or `rating_source`, and 214 of the 369 trainable labels sit on rows the
          profiler calls not-sit-down. The blind re-derivation the plan specified would have
          collapsed the training set to 157 — and every row it removes is rated 1–5, so it would
          have taken the low end the model learns from. See D-14.
    - [x] Sub-task: *Added:* skip any row with a `user_rating` or `rating_source`. 1,263 rows
          corrected, 216 human decisions preserved, training set unchanged at 369. Approved by the
          user against the measured alternative.
    - [x] Sub-task: Leave the 8,501 never-profiled rows alone; there is nothing to derive from.
- [x] Task: Validate the backfill — c609e34
    - [x] Sub-task: Every pillar column has more than one distinct value. Lowest is 2
          (`pillar_is_sit_down`, a BOOL); the two enums read 3 and 3; `match_score` reads 91.
    - [x] Sub-task: Spot-check a sample against the raw JSON. Stronger than a sample: 0 mismatches
          across all 2,766 rows on an INT64, a STRING-enum and a BOOL representative.
    - [x] Sub-task: Record per-column non-null coverage in `decision_log.md`. See D-14.
    - [x] Sub-task: Run on the snapshot copy first, compare, then prod. *Deviation:* rehearsed on a
          throwaway `fsa_master_rehearsal_20260923` (7-day expiry) instead. Backfilling
          `fsa_master_backup_20260923` would have destroyed the restore point it exists to be.
          Production then verified byte-identical to the validated rehearsal across all 19
          affected columns.

### Phase 5 checkpoint

**Manual verification.** `pytest app/ scripts/` — 241 passed (233 before the guard move). Python
3.11 parity checked with `ast.parse(feature_version=(3,11))` on every changed file. Production
figures after the run: 11,268 rows, 411 labels, 369 trainable, 0 surviving sentinels, 2,766 of
2,767 profiles extracted on all 14 paths, `in_scope` 10,693 → 9,465. Cross-table checks against the
pristine snapshot: 0 labels changed, 0 human-triaged rows moved.

**Deviations:** five, all recorded above and in D-14 — coverage tallies moved to `--validate`, the
`in_scope` human guard, the three-site guard move pulled forward from Phase 6, the Places writer
change, and rehearsing on a throwaway rather than the restore point.

**Cost:** £0. Five SQL `UPDATE`s over data already in the table; no Gemini, no Places, no new rows.

## Phase 6: Dual-Write

*Writers populate both representations so the deployed app keeps working mid-migration.*

- [x] Task: Populate the new columns going forward — 7d1bdd2
    - [x] Sub-task: `SCRIPT_MERGE_INSIGHTS` writes the typed columns and
          `gemini_profiled_at = CURRENT_TIMESTAMP()`, keeping `gemini_insights_structured` as the
          raw audit trail. **All 14 assignments generated from `PILLAR_FIELDS`, extracting from
          `S.gemini_insights` (the scratch table) — reading `T` would pick up the previous run's
          profile, since `T`'s own column is not written until this same statement.**
    - [x] Sub-task: The Places merge writes `maps_found` / `maps_lookup_at` and stops writing `-1`.
          Pulled forward into Phase 5 — 4cf568b.
    - [x] Sub-task: Switch the "needs Maps" guard from `maps_rating IS NULL` to
          `maps_lookup_at IS NULL`, preserving do-not-retry. Pulled forward into Phase 5 — 4cf568b.
          Three sites, not one; see the Phase 5 deviation note.
- [x] Task: Single source of truth for model features (D3) — ba61e86
    - [x] Sub-task: Build both the training `SELECT` and the `ML.PREDICT` subquery from the Phase 3
          canonical schema. **`app/core/model_features.py`; `feature_select_list()` plus
          `feature_source_clause()`, the latter not asked for — see the deviation below.**
    - [x] Sub-task: Test that fails if the two feature sets diverge. **`TestTrainServeParity` in
          `app/core/test_model_features.py`: the two generated strings must contain the identical
          fragment, and neither may carry a hand-written `JSON_EXTRACT_SCALAR`.**

- *Deviation:* the `FROM`/`LEFT JOIN` is shared as well as the `SELECT` list. The demographics join
  normalises the postcode (`REPLACE(UPPER(...), ' ', '')`); two copies of that rule could diverge
  and skew `lsoa`/`msoa`/`imd_rank` on exactly the rows whose postcodes are formatted
  inconsistently — the same silent-skew failure the D3 task exists to close, one line further down.
- *Deviation:* **the feature set grew from 6 to 8, and the `IFNULL(..., 0)` defaults are gone.**
  Neither was spelled out in the task, but both follow from "build it from the canonical schema":
  `is_feature` marks eight fields, and `pillar_geo_specificity` / `pillar_establishment_type` are
  enums the old SQL cast to INT64 while `pillar_is_sit_down` was not read at all. Dropping the
  defaults is R6 — an always-zero feature is indistinguishable from an uninformative one, which is
  precisely how D2 survived.
- *Deviation:* Phase 6 is **not independently deployable**, which the plan's merge rule assumes
  every phase boundary is. The live model's input schema is the old alias set
  (`score_1_value_and_volume_rating` … confirmed via `ML.FEATURE_INFO`: 19 features), so
  `ML.PREDICT` fails against it the moment this lands. **The retrain that Phase 9 schedules has to
  happen before this merges to `main`.** Priced below.
- *Deviation:* `scripts/recon_pipeline_state.py`'s `PRODUCTION_FEATURE_PATHS` is renamed
  `LEGACY_FLAT_FEATURE_PATHS`. Its two tests asserted that production reads those paths; they now
  assert the generated SQL does *not*, which turns a Phase 0 measurement into a D2 regression guard.
- *Known stale:* `tests/test_bqml_stress.py` still asserts `IFNULL(..., 0) → 0` against its own
  inline copy of the old SQL. It never touched the real query (that is why D2 survived it), it is
  mocked, and it is outside the CI set. Left for the Phase 12 reconciliation rather than edited
  here, but it now documents a pattern production has stopped using.

**Phase 6 checkpoint:** `pytest app/ scripts/` green at **266** tests (was 241; +16 in
`test_model_features.py`, +7 in `test_bq_scripts.py`, net +2 rewritten in the recon tests). All
edited files parse under `ast.parse(feature_version=(3,11))`. Three BigQuery **dry runs**, nothing
written: `CREATE MODEL` 5.7 MB, the `ML.PREDICT` input subquery 5.8 MB, the dual-write `MERGE`
11.7 MB. No Gemini call, no Places call, £0.

**Retrain, measured against production before running it** — the training population is 370 rows,
and the JIT pre-flight would fire **0 Places lookups, 0 postcode lookups, 7 Gemini calls** (the 7
labelled rows that have never been profiled). At $0.75/1M in and $3.75/1M out that is ≈ $0.02, with
grounding inside the 5,000/month free allowance; BQML training on 370 rows over a 5.7 MB scan is
pennies. **Under £0.05 all in.** 363 of the 370 already carry every typed pillar.

**Retrain executed, 2026-09-23 15:00–15:05 UTC** (approved). `ML.FEATURE_INFO` on the replaced model
reports **21 input features, up from 19**, and D2 is visible as repaired rather than merely fixed in
source:

| feature | before | after |
|---|---|---|
| `pillar_value_rating` | constant 0 | 0–8 |
| `pillar_community_score` | constant 0 | 0–6 |
| `pillar_linguistic_score` | constant 0 | 0–7 |
| `pillar_culinary_score` | constant 0 | 0–6 |
| `pillar_geo_specificity` | constant 0 | 3 categories |
| `pillar_is_sit_down` | not read | 2 categories |
| `pillar_establishment_type` | not read | 3 categories |
| `maps_rating` | min −1.0, 0 nulls | min 2.1, 166 nulls |

The last row is Phase 5 showing up in the model: the `-1` sentinel was a real value the tree could
split on, and it is now an honest NULL.

*Caution worth recording:* the first `ML.FEATURE_INFO` reading after the retrain returned the **old**
19 features. BigQuery served it from the 24-hour result cache, because the identical query had been
run before training. Any before/after measurement in this track must pass
`use_query_cache=False` or it will confirm whatever was true beforehand.

- *Deviation:* the retrain surfaced **D-16**, a new defect — a NULL postcode voided the profile
  prompt, so 7 labelled rows could never be profiled and were retried by every run. 126 unprofiled
  rows are affected. Fixed in e6ad77c along with a merge guard against recording a failed
  generation as a profile. The 7 orphaned `gemini_profiled_at` stamps were cleared (7 rows, £0);
  afterwards `profiled` = `have_json` = 2,767, stamped-without-profile = 0, labels 411.

**Phase 6 shipped.** Merged as 18f93a3; Cloud Build `9ca20503` SUCCESS; Cloud Run revision
`restaurants-fsa-00223-hxc` live. `ML.PREDICT` verified end-to-end against the retrained model on 5
rows (read-only, no MERGE): predictions 1.305–2.493, no schema error. CLAUDE.md's "live
inconsistency" and train/serve-parity warnings are deleted — both were true when written and are
not now.

## Phase 7: Switch Readers, Stale-Aware Refresh (R1)

- [x] Task: Read typed columns instead of parsing JSON — 0abf4e2
    - [x] Sub-task: Reduce `parse_insight_row` to reading columns; drop the V1 text branch. 0abf4e2
    - [x] Sub-task: The per-row `json.loads` in `enhance_dataframe_with_insights` disappears. 0abf4e2
    - [x] Sub-task: Update `DISPLAY_COLUMNS`. 0abf4e2
- [x] Task: Stale-aware refresh — 0abf4e2
    - [x] Sub-task: Threshold against `gemini_profiled_at` as a named constant. 0abf4e2
    - [x] Sub-task: The UI's "Estimated New Gemini Calls" uses the **same predicate** as the
          executor, so estimate and behaviour cannot drift apart again — that divergence is D1. 0abf4e2
    - [ ] Sub-task: The first stale sweep is budget-capped, with its £ cost stated before it runs.
          **Not run — and nothing is stale until 2027-03-22.** Priced below; needs a go-ahead.
    - [x] Sub-task: Tests for fresh / stale / never-profiled / forced. 0abf4e2
- [ ] Task: Conductor — User Manual Verification 'Rewire' (Protocol in workflow.md)

- *Deviation:* `parse_insight_row` is **deleted, not reduced.** Reducing it to "read fourteen
  columns off the row" left a function whose single caller no longer needed it —
  `enhance_dataframe_with_insights` is a frame-level column guarantee now, not a per-row map. Its
  three `insight_*` outputs went with it: `insight_authenticity` and `insight_vibe` read
  `cultural_authenticity_rating` / `atmosphere`, keys this profiler has never emitted, and
  `insight_verdict` / `insight_summary` / `detailed_insights` were computed for no reader at all.
- *Deviation:* the freshness predicate is a **new module**, `app/core/profile_freshness.py`, rather
  than a constant dropped into an existing one. The point of the sub-task is that the estimate and
  the spend are the same rule; a rule three files import is a place that rule can live, and
  `needs_gemini_profile(has_profile, profiled_at, force=, max_age_days=)` is what all three call.
- *Deviation:* the **training pre-flight** was brought onto the same predicate, which the task did
  not ask for. It passes `max_age_days=None` — fills gaps, never refreshes. It is the only
  scheduled caller, and a staleness rule there would spend money with nobody present; saying that
  in an argument is better than saying it in a comment beside a third hand-copied guard.
- *Deviation:* both find-queries now select `gemini_insights_structured IS NOT NULL AS has_profile`
  instead of the column. Nothing on either path parsed the JSON, and it is the widest column in the
  table.
- *Left for Phase 8 — done there, 0d02c8c, and it found [D-18].* `calculate_restaurant_priority` still reads
  `gemini_insights` / `gemini_insights_structured` for its staleness component rather than
  `match_score`. Switching it changes queue *ordering*, and this plan puts ordering changes behind
  a real-data check in Phase 8. Phase 10 has to revisit it anyway, when the legacy column goes.

**Phase 7 checkpoint:** `pytest app/ scripts/` green at **296** tests (was 268; +20 in
`test_profile_freshness.py`, +3 `test_ml_prediction.py`, +2 `test_train_bqml_model.py`, +3 net in
`test_data_processing.py`). All edited files parse under `ast.parse(feature_version=(3,11))`. Two
read-only BigQuery queries, `use_query_cache=False`, no writes, £0. Every name in `DISPLAY_COLUMNS`
was checked against the live table schema — all 44 resolve, so the grid cannot order a column that
is not there.

**Measured before choosing the threshold** (`fsa_master`, 2026-09-23):

| | rows |
|---|---|
| total | 11,268 |
| profiled (`gemini_insights_structured IS NOT NULL`) | 2,767 |
| profiled but unstamped | 0 |
| stamped without a profile | 0 |
| **stale at 180 days, today** | **0** |

The oldest `gemini_profiled_at` is the Phase 5 backfill, 2026-09-23 13:06 UTC. So switching
staleness on is a £0 change today and the first row becomes eligible on **2027-03-22**. The
"profiled but unstamped" count being 0 is what makes the predicate's money-safe default — unknown
age reads as *fresh* — unobservable in production rather than merely defensible.

**The sweep, priced, not run.** 8,501 rows carry no profile — the 1,116 the ledger sized is the
in-scope slice of that. Cost is not linear in the row count, because `_MODEL_PARAMS_STRUCT` enables
`tools: [{"googleSearch": {}}]` and the free allowance is 5,000 grounded calls a month:

| | tokens | grounded search | total |
|---|---|---|---|
| 1,116 in-scope rows | $7–$24 | inside the free allowance | **£6–£47** (the ledger's figure, unchanged) |
| all 8,501 unprofiled | $53–$183 | 3,501 over, at the ledger's $14/1,000 | **≈£80–£180** |

Nothing here is measured. The Phase 6 retrain's "≈$0.02 for 7 calls" was an a-priori figure from the
same $0.75/$3.75 rates, not a bill, so it is not independent evidence and the ranges do not narrow
because of it. Both are wide for the reason already recorded: grounded calls inject retrieved
results into the input and `gemini-3.8-flash` bills thinking tokens as output, neither of which is
visible in the stored `.result`. The 20-row pilot (~$0.15, reading `usageMetadata` off
`full_response`) is still what collapses the range. **0 of the 8,501 are labelled**, so either
sweep buys triage-queue coverage, not model quality. Needs an explicit go-ahead.

## Phase 8: Honest Missing Data and Free Coordinates (D4, R4) [checkpoint: 4aa719e]

- [x] Task: Missing location stops scoring as perfect — 0d02c8c
    - [x] Sub-task: `extract_outcode("")` stops returning `"SW16"`; return an explicit unknown. 0d02c8c
    - [x] Sub-task: Unknown-location rows get a low-or-neutral proximity score, not the maximum. 0d02c8c
    - [x] Sub-task: Update `app/core/test_scoring_priority.py`, which asserts the current default. 0d02c8c
    - [x] Sub-task: Confirm on real data that the top of the queue moves as expected. 0d02c8c
- [x] Task: Keep FSA coordinates at ingest — 0d02c8c
    - [x] Sub-task: Flatten the API's nested `Geocode` into `latitude`/`longitude` in
          `process_and_update_master_data`. 0d02c8c
    - [x] Sub-task: Add them to `ORIGINAL_COLUMNS_TO_KEEP`, which currently drops `Geocode`. 0d02c8c
    - [x] Sub-task: Remove the dead `Geocode.Latitude` handling from `write_to_bigquery`. 0d02c8c
    - [x] Sub-task: Test that ingest stores coordinates (the miss-path erasure is already fixed in
          Phase 1). 0d02c8c
- [x] Task: Purge sentinel awareness from readers — 0d02c8c
    - [x] Sub-task: UI metric, "Has Google Maps Rating" filter, and sorting read `maps_found`. 0d02c8c
    - [x] Sub-task: Confirm no feature column can receive `-1`. 0d02c8c
- [x] Task: Staleness reads the profile stamp, not the JSON blob (carried from Phase 7) — 0d02c8c
    - [x] Sub-task: Switch the component to `gemini_profiled_at`. 0d02c8c
    - [x] Sub-task: Measure the ordering change on the live table before keeping it. 0d02c8c
- [x] Task: Conductor — User Manual Verification 'Honest Missing Data' (Protocol in workflow.md)
    - Verification plan presented; the user chose to verify against the deployed app rather than a
      local Streamlit run, so the merge went first and the eyeball happens on Cloud Run.

- *Deviation:* **D4's headline turned out to be latent, not live.** `extract_outcode("")` really does
  return the anchor's own outcode, and that really would score a blank postcode as 0 km. No row
  reaches it: 126 postcodes are NULL and **0** are the empty string, and a NULL arrives in the frame
  as NaN, so the old `postcode or PostCode` read passed `'nan'` to the lookup, which fell through to
  the central-London fallback — 9.59 km, proximity 14.7. Two inventions rather than one. The fix
  stands as specified (10.0 and a blank distance for the 106 unplaceable rows), but the queue
  movement the task predicted comes from the staleness fix below, not from this.
- *Deviation:* `get_outcode_coordinates` **keeps its SW16 default**, and a new
  `lookup_outcode_coordinates` returns `Optional`. They are two different questions: the UI's anchor
  box has to resolve an empty string to somewhere, and a restaurant row must be allowed to have no
  known location. One function cannot answer both honestly.
- *Deviation:* the distance for an unplaceable row is **NaN, not 0.0**. Every sort in the UI already
  passes `na_position="last"` and `.mean()` skips NaN, so the grid shows a blank cell instead of a
  fabricated distance for free.
- *Deviation:* the Maps slicer gained a **fourth option**. "No Google Maps Rating" was one answer to
  two questions — 243 rows Places was asked about and had no match for, which must never be
  re-queried, and 8,762 nobody has asked about, which are the backlog. Keeping them in one bucket
  would have left the filter honest about `maps_found` and still useless for finding the backlog.
- *Deviation (new defect, [D-18]):* the staleness switch was expected to change nothing and changed
  1,051 rows. `gemini_insights or gemini_insights_structured` returns **NaN** whenever the V1 text is
  NULL, because NaN is truthy — so 10,152 rows read as never profiled, 1,020 of which hold a profile
  *and* a prediction and were being offered for re-profiling at maximum priority. The same trap was
  laid in the postcode read. `first_present()` now exists so it cannot be laid a third time.
- *Correction:* [D-17] recorded that `gemini_insights` is "NULL on every row". It is non-NULL on
  **1,116** — V1 text on rows with no V2 profile at all. The entry is corrected in place; the
  consequence is that the two D1 surfaces did not agree before Phase 7, they disagreed on those
  1,116 rows and always in the direction of under-reporting the bill.

**Phase 8 checkpoint:** `pytest app/ scripts/` green at **310** tests (was 296: +8 in
`test_scoring_priority.py`, +4 in `test_data_processing.py`, +2 in `test_st_app_unified_view.py`).
All edited files parse under `ast.parse(feature_version=(3,11))`. Five read-only BigQuery queries,
`use_query_cache=False`, no writes, £0. Before/after was measured by importing the pre-merge
`data_processing.py` from git alongside the new one and scoring the same 11,268 rows through both.

| | before | after |
|---|---|---|
| proximity score, 106 unplaceable rows | 14.7 (a fabricated 9.59 km) | **10.0**, distance blank |
| rows scored as never profiled | 10,152 | **8,501** (the true count) |
| staleness 100 on a row that holds a profile *and* a prediction | 1,020 rows | **0** |
| unprofiled rows in the top 25 / 100 / 500 | 0 / 7 / 178 | **4 / 37 / 315** |
| Maps backlog as the UI reports it | 9,005 ("no rating") | **8,762** never asked + 243 permanent misses |

The queue now puts unprofiled restaurants at the top of the list of restaurants to profile, which it
did not before. Nothing here spends money by itself — it changes which rows the next budgeted run
would spend it on.

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
