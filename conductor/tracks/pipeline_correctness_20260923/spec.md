# Specification: Pipeline Correctness & Simplification

## 1. Overview

An audit of the enrichment → scoring → prediction pipeline found several defects that cause
silent overspend, degrade model quality, and distort the priority queue. This track fixes them,
adds an evaluation harness so model quality is measurable rather than assumed, and removes four
superseded surfaces.

The work is sequenced so that the cheap, no-schema-change fixes land first and stop the bleeding,
before any migration touches `fsa_master`.

## 2. Defects found

### D1 — Every prediction run re-pays for Gemini (cost)
`app/services/ml_prediction.py:46` decides which restaurants need profiling with
`row.gemini_insights is None`. But `SCRIPT_MERGE_INSIGHTS` (`scripts/bq_scripts.py:161`) sets
`T.gemini_insights = NULL` on every successful enrichment. The column is therefore *always* NULL,
so every "Generate Predictions" click re-runs `AI.GENERATE` for every targeted restaurant — while
the UI's batch estimate (`st_app.py:704`, which correctly checks `gemini_insights_structured`)
reports most of them as cached. `scripts/train_bqml_model.py:39` already checks the right column,
so the two JIT paths disagree.

### D2 — Five of seven Gemini model features are constant zero (model quality)
The profiler prompt asks for nested JSON (`"1_value_and_volume": {"rating": 4}`), but both
`scripts/train_bqml_model.py:82-87` and `app/services/ml_prediction.py:103-108` read flat paths
(`$.1_value_and_volume_rating`). Those resolve to NULL and fall through to `IFNULL(..., 0)`. Only
`$.match_score` is a real top-level key. The Boosted Tree is effectively trained on Maps signals,
location, and `match_score` alone.

Note that correcting the paths alone is insufficient for pillar 4:
`4_geographic_precision.specificity_level` is an enum string (`"HYPER_LOCAL_CITY"`) wrapped in
`CAST(... AS INT64)`, so it would still resolve to 0.

**The nested shape is assumed, not observed** — it is read off the prompt text, and no artifact in
the repo records what `gemini_insights_structured` actually contains. See D14 and `decision_log.md`
D-01. Phase 0 settles it before any column is designed.

### D13 — A past migration mis-derived `in_scope` (data quality) — **CONFIRMED**
`scripts/migrate_to_in_scope_workflow.py:45` and `:70` — already run against production — gate
categorisation on
`JSON_EXTRACT_SCALAR(gemini_insights_structured, '$.6_establishment_integrity_is_sit_down_restaurant')`,
the same flat convention as D2. If that path resolves to NULL, those branches never fired and
`in_scope` was assigned from `maps_types` alone. This matters beyond tidiness: `in_scope` is the
predicate governing which restaurants get profiled at all.

**Phase 0 confirmed it**: that path resolves on **0** rows, and **1,476 of 2,766** profiled rows now
contradict their own profile. A second consequence surfaced in Phase 2 — the training query filters
on `in_scope`, so the defect also discards **42 of 411** hand-entered labels. See D-08 and D-10 in
`decision_log.md`.

### D14 — The profiler's output shape is unconstrained (root cause of D2)
`_MODEL_PARAMS_STRUCT` (`scripts/bq_scripts.py:104-127`) sets `generationConfig`, `safetySettings`
and `tools`, but there is **no `responseSchema` or `responseMimeType` anywhere in the repo**. Shape
is requested in prose only — which is why every consumer wraps the result in
`REGEXP_EXTRACT(..., r'(?s)[{{].*[}}]')`.

The recorded ADK eval outputs (`app/.adk/eval_history/*.json`) show what that produces across runs
of one prompt: pillar keys drift between `value_and_volume`, `pillar_1_value_volume`, and a
`pillars: {...}` wrapper, with inner keys varying across `{score, description,
anthropological_signal}`, `{score, analysis}`, and `{score, status, analysis}`. Those come from the
ADK agent, whose instruction is looser than the profiler's, so they establish risk rather than proof.

If the profiler drifts the same way, no fixed path set is safe and `gemini_insights_structured` is
heterogeneous *across rows* — making a fixed-column backfill produce coverage that depends on when
each row happened to be profiled. Correcting the paths without constraining generation leaves the
pipeline one model revision from the same breakage.

**Phase 0 measured this and the risk did not materialise**: the profiler is at 2,766/2,766
conformance on every required key, so the drift is an ADK-agent phenomenon, not a profiler one.
Phase 3 is downscoped accordingly — see D-08. The residual risk is real but is now addressed by
detection (a contract test and a merge-time conformance count) rather than by changing generation.

### D15 — `train_bqml_model.py --dry-run` can spend money
The JIT pre-flight block at `train_bqml_model.py:26-60` runs **before** `if dry_run:` is evaluated
and triggers Maps, Gemini and postcode enrichment for any labelled row missing that data — 7 rows
currently qualify for a grounded `AI.GENERATE`. `CLAUDE.md` documents the flag as "validate BQML
training SQL without spending". Deferred to Phase 11; see D-09.

The obvious fix is not drop-in: `tools: [{"googleSearch": {}}]` is enabled, and grounded search is
generally incompatible with constrained decoding on Gemini.

### D-note — the existing tests cannot catch this class of bug
`tests/test_bqml_stress.py` feeds its own flat literal into its own inline SQL, so it validates
`SAFE_CAST` and never touches the real query in `train_bqml_model.py`.
`app/core/test_data_processing.py::TestParseInsightRow` covers only the legacy V1 free-text branch —
the V2 nested branch (`data_processing.py:133-146`) has no test at all. `tests/eval/eval_config.yaml:23`
passes if *any* of a few top-level keys is present. A contract test running the real extraction
against a recorded payload is therefore part of the fix, not an extra.

### D3 — Feature lists are duplicated by hand (maintenance)
The training `SELECT` list and the `ML.PREDICT` subquery are copy-pasted across two files and must
be edited in lockstep or predictions silently skew from training. The same pillar schema is parsed
a third time in Python by `parse_insight_row`.

### D4 — Missing data scores as good data (ranking)
- `extract_outcode("")` returns `"SW16"` (`data_processing.py:219`) — the default anchor. A
  restaurant with a blank postcode gets distance 0 and the maximum proximity score, so absent data
  ranks it at the top of the queue.
- A Google Places miss writes `maps_rating = -1.0` (`enrich_maps_data.py:61`). That value counts
  toward the "Google Maps" metric, passes the "Has Google Maps Rating" filter, participates in
  sorting, and is fed to BQML as a genuine rating.

### D5 — A Places miss will erase FSA coordinates (data loss, latent)
`enrich_maps_data.py:61` writes `latitude: None, longitude: None` on a miss, and the MERGE
unconditionally overwrites both columns. This is harmless today only because nothing else ever
populates them — it becomes destructive the moment FSA coordinates are stored at ingest (R4).

### D6 — The weekly cron reads the entire master table (cost)
`app/cron/fetch_weekly.py:87` calls `load_master_data` → `load_all_data_from_bq`, which issues
`SELECT *` (`bq_utils.py:90`) and materialises every column of every row in memory purely to build
a set of FHRSIDs. Cost and memory grow without bound as the table does.

### D7 — Ingest pagination is unbounded
`fetch_data_for_all_coordinates` (`data_processing.py:28`) loops `while True`, breaking only when a
page returns fewer than `max_results`. An API that keeps returning exactly `max_results` never
terminates.

### D8 — Priority scoring is recomputed row-by-row on every rerun (latency)
`calculate_restaurant_priority` (`data_processing.py:301`) iterates with `.iterrows()` and performs
a `sorted()` over all 310 outcode keys *inside* the loop (`:254`). It runs on load and again in the
predictions tab on every Streamlit rerun, uncached.

### D9 — Failures present as empty results
Every BigQuery helper catches bare `Exception` and returns `[]` / `False`. An expired credential or
malformed query surfaces in the UI as "No data found matching criteria."

### D10 — Shared mutable temp tables
`execute_gemini_enrichment` `CREATE OR REPLACE`s `recents` and `genairesults_temp` in the production
dataset with no expiration; a JIT enrichment during training collides with a UI batch run. The
`bulk_update_reviews` temp table is deleted after `job.result()` with no `finally`, so it leaks
whenever the MERGE raises (`bq_utils.py:214`).

### D11 — Test suite cannot be run as a whole
`tests/` carries no markers, so a bare `pytest` attempts live Vertex calls, real BigQuery, and binds
port 8000. Only `pytest app/` is offline-safe, and that is all Cloud Build runs.

### D12 — Two dependency sources, two Python versions
Unpinned `requirements.txt` drives Docker and Cloud Build (Python 3.11); `pyproject.toml` + `uv.lock`
drive local development (Python 3.13). This gap produced the recurring f-string-backslash build
failures visible in the history.

## 3. Decisions taken

| # | Decision | Rationale |
|---|---|---|
| R1 | Gemini profiles refresh on a **staleness threshold**, not once-only | Makes the "Model Refresh (Stale Re-scoring)" preset refresh the profile rather than re-running `ML.PREDICT` over unchanged features |
| R2 | Pillars are parsed into **typed BigQuery columns** at enrichment time | Removes hand-written JSON parsing from three sites and resolves D2 and D3 together |
| R3 | Missing data is **represented as missing** | Unknown location must not outrank known-nearby; a failed lookup is not a rating |
| R4 | FSA `Geocode` is **kept at ingest** as the base location; Places overwrites when it finds a match | Every restaurant gets real coordinates for free; centroid fallback becomes the exception |
| R5 | `manual_review`, `gemini_insights`, the sidebar BigQuery-path input, and `app/maps_agent/` are **removed** | All four are superseded or non-functional |
| R6 | Migrations are **additive first**, run against a **dated snapshot** of `fsa_master`, via `--dry-run`-capable scripts executed only on explicit approval | `fsa_master` holds the only copy of the hand-entered ratings |
| R7 | The model gets a **held-out evaluation against a `match_score` baseline** | At 50–200 labels a Boosted Tree may not beat ranking by `match_score` alone; today there is no way to tell |

### R1 detail — staleness needs a new timestamp
No column records when a Gemini profile was *written*. `predicted_at` is set by `ML.PREDICT`, which
is a different event. A `gemini_profiled_at TIMESTAMP` column must be added and set by the merge
step. Rows profiled before this track have unknown age and are treated as stale once, then tracked
normally thereafter.

### R3 detail — the sentinel carries a second meaning
`-1` currently doubles as the "already looked up, do not retry" marker, via the
`AND maps_rating IS NULL AND maps_reviews IS NULL` guard at `enrich_maps_data.py:26`. It cannot
simply be nulled: an explicit `maps_lookup_at` timestamp must take over that role, or every run will
re-query Places for permanent misses.

## 4. Target schema additions

> **CONFIRMED** by the Phase 0 key census, 2026-09-23. Convention A is what the column actually
> holds: all 17 source paths below resolve on 2766 of the 2767 profiled rows (the 1 exception is a
> single unparseable payload), and every profiled row shares one identical top-level key set. No
> alias `COALESCE` is needed in the Phase 5 backfill. See D-08 in `decision_log.md`.

All additive. Populated by the enrichment merge; backfilled once from existing
`gemini_insights_structured` values.

| Column | Type | Source |
|---|---|---|
| `gemini_profiled_at` | TIMESTAMP | set by `SCRIPT_MERGE_INSIGHTS` |
| `maps_lookup_at` | TIMESTAMP | set by the Places merge, hit or miss |
| `maps_found` | BOOL | false on a Places miss |
| `match_score` | INT64 | `$.match_score` |
| `pillar_value_rating` | INT64 | `$.1_value_and_volume.rating` |
| `pillar_value_verdict` | STRING | `$.1_value_and_volume.verdict` |
| `pillar_community_score` | INT64 | `$.2_demographic_community.score` |
| `pillar_community_evidence` | STRING | `$.2_demographic_community.evidence` |
| `pillar_linguistic_score` | INT64 | `$.3_linguistic_signal.score` |
| `pillar_linguistic_menu_type` | STRING | `$.3_linguistic_signal.menu_type` |
| `pillar_geo_region` | STRING | `$.4_geographic_precision.region_identified` |
| `pillar_geo_specificity` | STRING | `$.4_geographic_precision.specificity_level` (categorical, **not** INT64) |
| `pillar_culinary_score` | INT64 | `$.5_culinary_uncompromisingness.score` |
| `pillar_culinary_pander_check` | STRING | `$.5_culinary_uncompromisingness.pander_check` |
| `pillar_is_sit_down` | BOOL | `$.6_establishment_integrity.is_sit_down_restaurant` |
| `pillar_establishment_type` | STRING | `$.6_establishment_integrity.type` (categorical) |
| `summary_reasoning` | STRING | `$.summary_reasoning` |

**Model features** become: `match_score`, the four pillar integer scores, `pillar_geo_specificity`,
`pillar_is_sit_down`, `pillar_establishment_type`, plus the existing Maps/location/demographic
fields. The free-text pillar columns are display-only and must stay out of the feature list.

`gemini_insights_structured` is retained as the raw audit trail; it stops being parsed at read time.

## 5. Acceptance criteria

*Walked one by one on 2026-09-24 and recorded in D-30. Each tick names its evidence; the two that
are not ticked say why rather than being left silently blank.*

- [ ] A second consecutive "Generate Predictions" run over the same selection issues **zero**
      `AI.GENERATE` calls, and the UI's "Estimated New Gemini Calls" matches what actually runs.
      — *Pending the live two-run check (Part A). The estimate and the spend already call the same
      predicate, `needs_gemini_profile`, from `count_needing_gemini_profile` and
      `generate_predictions`; what is outstanding is the observation, not the wiring.*
- [x] Profiles older than the staleness threshold are refreshed; fresher ones are not.
      — `app/core/test_profile_freshness.py`, 20 tests. **Not observable in production:** 0 rows are
      stale at the 180-day threshold and the first becomes eligible 2027-03-22.
- [ ] New profiles conform to a single canonical output shape, verified on a sample — the shape is
      guaranteed by configuration, not requested in prose (D14).
      — **Deliberately not met; superseded by D-08.** Constrained decoding is unavailable alongside
      `googleSearch` grounding, and dropping grounding stayed off the table. Recon measured
      2,766/2,766 nested paths resolving, so the prose example already holds the shape and the
      two-step rework bought nothing. What shipped instead is detection: one definition in
      `pillar_schema.py`, the contract test below, and `sql_conformance_check` at merge time.
- [x] A contract test runs the **real** extraction SQL against a payload recorded from production,
      so a path/shape mismatch fails the build rather than silently zeroing a feature.
      — `app/core/test_pillar_schema.py` against 10 payloads recorded from production in
      `tests/fixtures/gemini_profiles/` (8 clean, 1 markdown-fenced, 1 unparseable). The parser and
      the extraction SQL are both generated from `PILLAR_FIELDS`, and
      `test_generated_sql_reads_the_nested_paths_not_the_flat_ones` pins the SQL to the same paths
      the fixtures are parsed with. The SQL's own live reading is `sql_conformance_check`:
      2,766/2,767 conform on all 14 paths.
- [x] Every pillar feature reaching BQML has a non-degenerate distribution — verified by a
      `SELECT COUNT(DISTINCT ...)` check, not by inspection.
      — Built by `recon_pipeline_state.py:91` and `backfill_pillar_columns.py:188`. Measured:
      match_score 91, value 20, community 22, linguistic 17, culinary 23, geo_specificity 3,
      is_sit_down 2, establishment_type 3 — every one ≥ 2. Before the fix four of them were
      constant 0 *inside the trained model*.
- [x] The training feature list and the `ML.PREDICT` feature list derive from a single definition;
      a test fails if they diverge.
      — `app/core/test_model_features.py::TestTrainServeParity`. Confirmed against the served model:
      21/21 exact match with `feature_select_list()`, label excluded.
- [x] A restaurant with a blank or unmappable postcode does not outrank a known-nearby restaurant.
      — `test_scoring_priority.py`: `test_a_missing_postcode_is_unknown_not_sw16`,
      `test_an_unknown_location_does_not_score_as_perfect_proximity`,
      `test_an_unknown_location_still_beats_nothing`,
      `test_a_nan_postcode_is_unknown_rather_than_the_string_nan`. Live: the 106 unplaceable rows
      went from 14.7 (scored as Trafalgar Square, 9.59 km) to 10.0 with a blank distance, and they
      are exactly the rows whose proximity changed.
- [x] No row has `maps_rating = -1`; UI counts, filters, and sorting reflect true Maps coverage;
      permanent Places misses are still not re-queried.
      — 0 surviving sentinels in `maps_rating`, `maps_reviews`, `price_level` and `match_score`.
      Coverage is now three states rather than two: `maps_found` TRUE 2,263 / FALSE 243 / NULL 8,762.
      Re-query guard: `test_a_permanent_miss_is_not_re_queried` and
      `test_a_miss_records_the_lookup_instead_of_a_sentinel`; the same `maps_lookup_at IS NULL`
      predicate now gates all three call sites.
- [x] Newly ingested restaurants have `latitude`/`longitude` before any Places call, and a Places
      miss does not clear them.
      — `test_the_fsa_coordinates_are_kept_at_ingest`,
      `test_ingest_coordinates_are_numbers_not_the_api_strings`,
      `test_places_miss_does_not_erase_existing_coordinates`,
      `test_hit_without_location_preserves_coordinates`.
- [x] Held-out evaluation reports the model's error against a `match_score`-ranking baseline, with a
      documented recommendation on whether to keep the BQML path.
      — Phase 9, 292 train / 77 holdout on `FARM_FINGERPRINT(fhrsid) MOD 5`. Tree 0.737 MAE / ρ
      0.689 against `match_score` 0.679 / 0.645; paired bootstrap ΔMAE +0.058, 95% CI
      [−0.073, +0.210]. Recommendation recorded and reasoned: **keep BQML, do not trust it above
      `match_score` yet**, re-measure at ~600 in-scope labels.
- [x] `manual_review`, `gemini_insights`, the path input, and `app/maps_agent/` are gone with no
      dangling references.
      — Columns dropped: `fsa_master` is 42 columns, from 44. `app/maps_agent/` has zero references
      outside `conductor/`. The BigQuery path is the `DEFAULT_BQ_PATH` constant, no longer a text
      box. The surviving mentions of the two column names are a spent migration script that
      documents itself as unrunnable, and tests asserting their absence.
- [x] `pytest` at the repo root runs only offline tests; live tests run behind an explicit marker.
      — `addopts = "-m 'not integration'"` (D11): 469 passed, 10 deselected. `pytest -m integration`
      runs the 10 deliberately; `pytest -m ""` runs everything.
- [x] `pytest app/` stays green throughout; Cloud Build passes on Python 3.11.
      — Cloud Build `3ca778b8`, Python 3.11.16, 457 collected / 10 deselected / 447 selected, all
      green, deployed as revision `restaurants-fsa-00232-hxh`. Until D12 the CI gate had never run
      `tests/` at all.

## 6. Out of scope

- Retiring BQML in favour of `match_score` ranking — Phase 7 produces the evidence; the decision is
  a separate track.
- Rewriting SQL construction to use query parameters. Worth doing (it would also end the f-string
  backslash fragility), but it touches every query and is better isolated.
- Multi-user support, authentication, or making the BigQuery path genuinely configurable.
- ~~Any change to the profiler prompt's wording or the 6-pillar rubric itself.~~
  **Amended 2026-09-23** — constraining the *output shape* (response schema / model params, and
  possibly the `googleSearch` tool) is now **in scope** as Phase 3, because D14 identifies it as the
  root cause of D2. The 6-pillar rubric and the prompt's analytical wording remain out of scope:
  this is about guaranteeing the envelope, not changing what is asked for. See `decision_log.md` D-02.
