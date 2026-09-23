# Decision Log: Pipeline Correctness & Simplification

**Date:** 2026-09-23
**Track:** pipeline_correctness_20260923

---

## D-01 — Evidence precedes design: recon before the schema migration

**Date:** 2026-09-23

### Context
The first version of this plan specified a 17-column additive migration whose backfill read
**nested** JSON paths (`$.1_value_and_volume.rating`) out of `gemini_insights_structured`. That
shape came from reading the profiler prompt in `scripts/bq_scripts.py`, not from any observed
output.

A review found **four mutually inconsistent key conventions** for the same six pillars in the repo:

| Convention | Shape | Where |
|---|---|---|
| A — nested, numeric prefix | `"1_value_and_volume": {"rating", "verdict"}` | the prompt's own `### EXAMPLE OUTPUT`, `scripts/bq_scripts.py:63-101` |
| B — flat, numeric prefix | `$.1_value_and_volume_rating` | `train_bqml_model.py:82-87`, `ml_prediction.py:103-108`, `migrate_to_in_scope_workflow.py:45,70` |
| C — flat DataFrame columns | `res["1_value_and_volume_rating"]`, written from nested reads | `data_processing.py:134-135`, `st_app.py:28` |
| D — nested, no prefix, different leaf names | `"value_and_volume": {"score", "description", "anthropological_signal"}` | recorded output, `app/.adk/eval_history/*.json` |

No artifact anywhere in the repo records what the column actually contains. `CLAUDE.md:95-98`
already concedes this and says to verify against real rows first.

### Decision
Phase 0 is a read-only reconnaissance gate. Nothing from Phase 3 on is designed until a full key
census over real rows has settled the shape. The §4 column table in `spec.md` is **provisional**
until then.

### Reasoning
The failure mode is asymmetric. If the assumed shape is right, recon costs pennies and an hour. If
it is wrong, the migration, backfill and retrain all silently produce null columns — the exact
defect (D2) this track exists to fix, reintroduced one layer down.

---

## D-02 — D2 is a symptom; the root cause is unconstrained output (D14)

**Date:** 2026-09-23

### Context
`_MODEL_PARAMS_STRUCT` (`scripts/bq_scripts.py:104-127`) sets `generationConfig`, `safetySettings`
and `tools`, but there is **no `responseSchema` or `responseMimeType` anywhere in the repo**. The
output shape is requested in prose only, which is why every consumer wraps the result in
`REGEXP_EXTRACT(..., r'(?s)[{{].*[}}]')`.

The recorded ADK eval outputs show what unconstrained generation does to key names across runs of
the same prompt: `value_and_volume` / `pillar_1_value_volume` / a `pillars: {...}` wrapper, with
inner keys varying across `{score, description, anthropological_signal}`, `{score, analysis}`,
`{score, status, analysis}`.

Those recordings come from the ADK agent, whose instruction is much looser than the profiler's
(which has an explicit example output and a "strictly follow this structure" schema reference), so
they establish *risk*, not that the profiler itself drifts.

### Decision
Treat the generation shape as the root cause and fix it inside this track (Phase 3), before the
typed columns are designed. **This supersedes the original `spec.md` §6 exclusion** on changes to
the profiler prompt and model params.

Mechanism to be chosen on the recon evidence, with costs presented first:
- **Recommended — two-step.** Keep the grounded call, then a cheap *ungrounded* `AI.GENERATE` with
  `responseSchema` that normalises the result into the canonical shape. Costs one extra call per
  profile but preserves the web grounding the "Culinary Anthropologist" prompt depends on.
- **Alternative — drop `googleSearch`** so constrained decoding becomes available directly. Cheaper,
  but removes grounding; the Phase 2 harness would have to quantify the quality regression first.

### Reasoning
`tools: [{"googleSearch": {}}]` is enabled, and on Gemini grounded search is generally incompatible
with constrained decoding — so `responseSchema` is not a drop-in addition, and the choice is a real
tradeoff rather than a config tweak. Correcting the JSON paths alone leaves the pipeline one model
revision away from the same breakage.

---

## D-03 — Existing profiles are backfilled as fresh, not stale

**Date:** 2026-09-23

### Context
The first plan set `gemini_profiled_at = NULL` for every pre-existing profile so they would "refresh
once, then track normally". Since the refresh predicate treats NULL as stale, that schedules a
re-profile of every already-profiled row in the table.

### Decision
Backfill `gemini_profiled_at` to `CURRENT_TIMESTAMP()` for rows that already have a profile,
following the precedent in `scripts/migrate_predicted_at.py:19-26`. Existing profiles count as
fresh; the first stale sweep is then a normal budget-capped operation with a £ estimate presented
before it runs.

### Reasoning
The headline defect of this track (D1) is paying twice for Gemini. A plan that fixes D1 and then
silently triggers a full-table re-profile has not reduced spend. `migrate_predicted_at.py` already
solved the identical problem for `predicted_at` and is the repo-native pattern.

---

## D-04 — The model is baselined before it is repaired

**Date:** 2026-09-23

### Context
The first plan recorded `ML.TRAINING_INFO` in Phase 0 and ran a held-out evaluation in Phase 7.
Training loss and held-out error are not comparable, so the plan could not have answered whether
fixing the features actually improved anything.

### Decision
Phase 2 builds the held-out harness and records two numbers *before* any feature repair: the current
model's MAE/RMSE, and a no-ML baseline ranking the same held-out rows by `match_score` alone. Phase
9 re-runs the identical harness and reports the delta.

### Reasoning
At 50–200 labels a Boosted Tree may not beat `match_score` ranking at all. That is a legitimate
outcome, but only if it is measured against a before-number produced by the same method. Moving the
measurement early also means the repair work is informed by whether the model is worth repairing.

---

## D-05 — A past migration may have mis-derived `in_scope` (D13)

**Date:** 2026-09-23

### Context
`scripts/migrate_to_in_scope_workflow.py:45` and `:70` — already run against production — gate
categorisation on
`JSON_EXTRACT_SCALAR(gemini_insights_structured, '$.6_establishment_integrity_is_sit_down_restaurant')`,
convention B above. If that path resolves to NULL, those branches never fired and `in_scope` was
assigned from `maps_types` alone.

### Decision
Recon measures whether that path ever resolves. If it does not, Phase 5 re-derives `in_scope` from
the now-typed `pillar_is_sit_down` for the affected rows, dry-run and diff-counted first.

### Reasoning
`in_scope` is not cosmetic — it is the predicate governing which restaurants get profiled at all, so
a silent mis-derivation propagates into what the pipeline spends money on.

---

## D-06 — The snapshot is the only non-production environment

**Date:** 2026-09-23

### Context
A sweep for a dev/staging project, dataset, or service found none. There is one project, one dataset
(`filipegracio_fsa_restaurants`), one master table, one Cloud Run service; even the scratch tables
`recents` and `genairesults_temp` are created in the production dataset. `fsa_master` holds the only
copy of the hand-entered `user_rating` labels, and there is no backup mechanism in code.

### Decision
`fsa_master_backup_20260923` serves as both the restore point and the rehearsal target: the Phase 4
migration and Phase 5 backfill run against it first, are validated there, and only then run against
production. The documented restore is
`CREATE OR REPLACE TABLE fsa_master AS SELECT * FROM fsa_master_backup_20260923`, verifiable against
the row count and `COUNT(user_rating)` recorded at snapshot time.

### Reasoning
A snapshot with no rehearsal and no written restore procedure is a backup in name only. Since no
separate environment exists, the snapshot has to serve both roles.

---

## D-07 — The auto-deploy trigger is confirmed, and it is region-scoped

*Recorded 2026-09-23, at the Phase 1 checkpoint.*

### Context

The plan flagged that `CLAUDE.md:36` and `README.md:62` both claim a Cloud Build trigger deploys on
every push to `main`, but that the trigger config is server-side and unverifiable from the repo. The
whole merge strategy — merge only at green phase boundaries — rests on that claim being true.

A plain `gcloud builds triggers list` returns five triggers, **none of them for this repo**. That
reading is wrong: the command defaults to the `global` region. The trigger is in `europe-west2`.

### Decision

The docs are accurate. Verified:

| Field | Value |
|---|---|
| Trigger | `restaurants-fsa-github` |
| Region | `europe-west2` (**not** `global` — must be passed explicitly) |
| Repository | `graciofilipe/restaurants_FSA` via the `filipegraciogithublondonconnection` GitHub connection |
| Event | push to `^main$` |
| Build file | `cloudbuild.yaml` |
| `includedFiles` / `ignoredFiles` | none — every push builds |
| `disabled` | not set |

Consequences, both now load-bearing for this track:

1. **Any merge to `main` deploys to Cloud Run.** Merge only at green, independently deployable phase
   boundaries.
2. **`cloudbuild.yaml:11` is a real gate, not documentation.** `python -m pytest app/ scripts/`
   fails the build before the image is pushed, so a broken test blocks the deploy rather than
   shipping alongside it. This is why Phase 1 added `scripts/` to that command.

### Reasoning

Recorded chiefly for the region trap: a future check that omits `--region=europe-west2` will
conclude no trigger exists and that `main` is safe to merge into freely. It is not.

---

## D-08 — Recon settles the shape: convention A, stable, no re-profile needed

*Recorded 2026-09-23, from `scripts/recon_pipeline_state.py` over all 11,268 rows. Total query cost
33.4 MiB scanned, about £0.0002.*

### Context

D-01 made Phase 0 a gate: nothing from Phase 3 on would be designed until a key census over real
rows settled which of the four conventions `gemini_insights_structured` actually uses. It has run.

### Findings

**1. The shape is convention A, and it is stable.** 2,766 of the 2,767 profiled rows share one
identical top-level key set:

```
["1_value_and_volume", "2_demographic_community", "3_linguistic_signal",
 "4_geographic_precision", "5_culinary_uncompromisingness", "6_establishment_integrity",
 "match_score", "summary_reasoning"]
```

The one exception is a single row whose payload contains no JSON object at all. All 17 second-level
paths in spec §4 resolve on all 2,766 — including the leaf names (`verdict`, `evidence`,
`menu_type`, `region_identified`, `specificity_level`, `pander_check`, `is_sit_down_restaurant`,
`type`). **Spec §4 is confirmed unchanged and is no longer provisional.**

Key drift exists but is confined to pillars 4 and 6, and only as *extra* keys alongside the required
ones — `6_establishment_integrity.note` on 14 rows, `.details` on 9, `.summary_reasoning` on 6, then
a long tail of 1–2 row variants (`status`, `summary`, `warning`, `integrity_violation`). Nothing the
schema reads is ever missing.

**2. D2 confirmed, at full severity.** Every one of the five flat pillar paths the production SQL
reads resolves on **zero** rows:

| Path read by `train_bqml_model.py` / `ml_prediction.py` | Non-null rows | Distinct values |
|---|---|---|
| `$.1_value_and_volume_rating` | **0** | 0 |
| `$.2_demographic_community_score` | **0** | 0 |
| `$.3_linguistic_signal_score` | **0** | 0 |
| `$.4_geographic_precision_specificity_level` | **0** | 0 |
| `$.5_culinary_uncompromisingness_score` | **0** | 0 |
| `$.match_score` | 2,766 | 91 |

Against the same rows, the nested equivalents resolve 2,766/2,766 with 20, 22, 17, 3 and 23 distinct
values respectively. **The deployed model has one real Gemini feature — `match_score` — and five
constant zeros.** `IFNULL(…, 0)` is why this never surfaced as an error.

**3. D13 confirmed and sized.** `'$.6_establishment_integrity_is_sit_down_restaurant'` resolves on
**0** rows; the nested form resolves on 2,766. The categorisation branches in
`migrate_to_in_scope_workflow.py` therefore never fired, and `in_scope` was assigned from
`maps_types` alone. The disagreement against the profiles:

| | Rows |
|---|---|
| `in_scope = TRUE` but the profile says **not** a sit-down restaurant | **1,458** |
| `in_scope = FALSE` but the profile says it **is** a sit-down restaurant | 18 |

1,476 of the 2,766 profiled rows are mis-categorised — over half. `in_scope` is true on 10,693 of
11,268 rows overall, so the column is doing almost no filtering work, which is exactly what makes
the Gemini spend broad.

**4. D1 is worse than the plan assumed.** The plan expected
`COUNT(*) WHERE gemini_insights IS NOT NULL` to be **0**. It is **1,116** — and the V1 and V2
columns are perfectly disjoint (`v1_and_v2 = 0`). Those 1,116 rows are legacy V1-only profiles that
never got a V2 structured profile. So the old guard at `ml_prediction.py:46` was wrong in *both*
directions: it skipped the 1,116 rows that most needed profiling, and re-profiled all 2,767 that
already had one. The Phase 1 fix (test `gemini_insights_structured`) is correct and now also means
those 1,116 rows become eligible — a cost item for Phase 7's budget-capped sweep, not free.

**5. The label set is 411, not 50–200.** 404 of them are on profiled rows, so the usable training
set is 404. Distribution is heavily skewed low:

| `user_rating` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| rows | 188 | 106 | 35 | 17 | 11 | 27 | 16 | 10 | 1 |

### Decisions that follow

- **Spec §4 stands as written.** The 17 columns and all 17 source paths are confirmed. Phase 5's
  backfill is a straight nested-path extraction with **no alias `COALESCE`** — D-01's main worry
  does not apply.
- **No legacy re-profile.** Every existing profile already carries every field the typed columns
  need. Phase 5 backfills from SQL alone. The "re-profile legacy rows into the canonical shape"
  option in Phase 5 is **withdrawn** — it would cost a full Gemini sweep to buy nothing.
- **Phase 3 (D14) is downscoped.** The drift the recorded ADK outputs showed does **not** appear in
  the profiler: 2,766/2,766 conformance on the required keys is strong evidence that the profiler's
  explicit example output already holds the shape. The two-step constrained-decode rework in D-02 is
  not justified by this evidence. Phase 3 becomes: define the canonical schema once in code, add the
  contract test against the Phase 0 fixtures, and add a conformance check at merge time so drift is
  *detected* rather than silently absorbed. Dropping `googleSearch` stays off the table.
- **Phase 5 gains a D13 re-derivation** of `in_scope` from `pillar_is_sit_down` for the 1,476
  disagreeing rows, dry-run and diff-counted first. The direction matters: re-deriving will move
  ~1,458 rows *out* of scope, which reduces future Gemini spend.
- **The stale-sweep budget must account for the 1,116 V1-only rows**, which Phase 1's fix newly
  exposes as unprofiled.

### Reasoning

D-01's bet paid off in the opposite direction from the one it feared: the shape is *more* reliable
than assumed, which removes work (no re-profile, no alias handling, a much smaller Phase 3) rather
than adding it. The findings that did land — D2 at full severity, D13 at 1,476 rows, D1's 1,116
hidden unprofiled rows — are all things the original plan would have got wrong by assumption.

---

## D-09 — `train_bqml_model.py --dry-run` can spend money (new defect, D15)

*Found 2026-09-23 while building the Phase 2 harness.*

### Context

`CLAUDE.md` documents `python -m scripts.train_bqml_model --dry-run` as "validate BQML training SQL
without spending". It does not do that. The JIT pre-flight block at `train_bqml_model.py:26-60` runs
**unconditionally, before** `if dry_run:` is ever evaluated, and it calls
`enrich_maps_data.enrich_restaurants_by_fhrsid`, `bq_utils.execute_gemini_enrichment` and
`enrich_postcode_demographics.enrich_postcodes` for any labelled row missing that data.

There are currently 7 labelled rows with no `gemini_insights_structured`, so a `--dry-run` today
would issue 7 grounded `AI.GENERATE` calls and an unbounded number of Places lookups.

### Decision

Recorded as **D15** and deferred to Phase 11, not fixed here — Phase 2 must not change the thing it
is baselining. The fix is to move the JIT block inside the non-dry-run branch.

In the meantime the Phase 2 harness never invokes `train_model()`. It imports
`build_training_select` and validates the SQL directly, which is why the refactor extracting that
function was worth doing rather than hand-copying the feature list a third time.

### Reasoning

This is the same failure shape as D1: a guard that reads as a cost control but is evaluated after
the spend, or on the wrong column. Worth naming separately because the *documentation* actively
misleads here — a flag called `--dry-run` is the last place anyone would look for a Gemini bill.

---

## D-10 — The `in_scope` filter discards 42 hand-entered labels

*Measured 2026-09-23 while sizing the Phase 2 split.*

### Context

`build_training_select`'s `WHERE` clause is `(m.in_scope = TRUE OR m.in_scope IS NULL) AND
m.user_rating IS NOT NULL`. Of 411 labelled rows, only **369** satisfy it; 42 are dropped for
`in_scope = FALSE`. Given D-08 showed `in_scope` is mis-derived on 1,476 of 2,766 profiled rows,
that is 10% of the hand-entered training data being discarded by a column known to be wrong.

Measured before assuming the worst: the 42 excluded rows have a mean `user_rating` of **1.21**
against **2.55** for the included ones, and only **2** of them carry a profile saying they *are* a
sit-down restaurant. The exclusion is therefore substantially correct in effect — these really are
the places the user rates lowest — and only 2 rows are clearly wrongly dropped.

### Decision

**No change in Phase 2.** The baseline must train on exactly what production trains on, or the
Phase 9 delta measures the change in row selection rather than the change in features.

Revisit in Phase 9, once the D13 re-derivation has corrected `in_scope`: at that point re-sizing the
training set is a real question, and worth asking whether 42 unambiguous negatives are data the
model should see rather than data to filter out.

### Reasoning

Two effects were tangled here and had to be separated: the label count is smaller than the recon's
411 suggested, but the cause is mostly legitimate filtering rather than the D13 defect. Acting on
the headline number without measuring the composition would have been a change made on a
misreading.

---

## D-11 — Baseline: the BQML model is beaten by the single feature it wraps

*Measured 2026-09-23 by `scripts/evaluate_model.py`. 292 training rows, 77 held-out, split by
`FARM_FINGERPRINT(fhrsid) MOD 5`. Phase 9 must reuse `--holdout_modulus 5` verbatim.*

### The three numbers

| Predictor | Features | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|---|
| `BOOSTED_TREE_REGRESSOR`, current features | ~20 | **0.847** | 1.283 | 0.545 | 0.545 |
| `match_score` alone (`LINEAR_REG`) | 1 | **0.697** | 1.095 | 0.668 | 0.603 |
| Training mean | 0 | 1.566 | 1.926 | 0.0 | n/a |

**The one-feature baseline beats the twenty-feature boosted tree on every metric** — 18% lower MAE,
15% lower RMSE, higher R², and a better ranking correlation, which is the property the app actually
depends on since the queue is sorted by prediction.

Both comfortably beat the mean, so `match_score` carries real signal. The tree then destroys part of
it.

### Why this is the expected result, not an anomaly

Five of the model's six Gemini features are pinned to `0` by D2. What remains beyond `match_score`
is largely high-cardinality categoricals — `postcode`, `localauthorityname`, `lsoa`, `msoa`,
`maps_types_array` — which give a boosted tree ample room to memorise 292 rows. Small data plus wide
categoricals plus five dead columns is a recipe for exactly this.

### Decision

Recorded as the Phase 2 floor; **no action taken now**. Phase 9 re-runs this harness unchanged and
reports the delta. Three outcomes are now possible and all are legitimate:

1. Repaired features push the tree past 0.697 MAE — the model earns its place.
2. It improves but still trails `match_score` — ship the linear baseline and retire BQML.
3. It does not improve — the pillars carry no signal the label responds to, and the profiler prompt,
   not the plumbing, is the thing to revisit.

The keep-or-retire recommendation belongs to Phase 9; this phase exists so that recommendation can
be made on numbers.

### Caveat recorded deliberately

The deployed `restaurant_preference_model` is **not** the model measured here. It was trained on all
369 in-scope labelled rows including these 77, so its own `ML.EVALUATE` scores data it memorised and
cannot serve as a baseline. The row above is a fresh model of the same type and features, trained on
the training split only. That is the honest comparison and the one Phase 9 will repeat.

---

## D-12 — The conformance check is advisory, and its first reading confirms D-08

**Date:** 2026-09-23 · **Phase:** 3 · **Status:** decided

### The mechanism

`app/core/pillar_schema.py` is now the single definition of the profiler's output: 14 JSON-derived
fields plus 3 non-JSON columns, each carrying its column name, BigQuery type, and key path. The
extraction SQL, the Python parser, the model feature list, and the conformance query are all
generated from that one tuple. `app/core/test_pillar_schema.py` runs the real extraction against the
real payloads captured in Phase 0 — the first test in this repo that could have caught D2.

`log_insight_conformance` (`app/services/bq_utils.py`) runs the generated check against the scratch
insights table immediately before the merge, and logs the per-path miss counts.

### First production reading, 2026-09-23

Run against `fsa_master` with `where='gemini_insights_structured IS NOT NULL'`:

| | |
|---|---|
| Profiles examined | **2,767** |
| Conforming on all 14 canonical paths | **2,766** |
| Unparseable (no JSON object at all) | **1** — FHRSID 1855447, a leaked reasoning trace |
| Rows missing any individual path while otherwise parseable | **0** |

Every one of the 14 `missing_*` counters reads exactly 1, and that 1 is the same row in all 14. So
there is no partial drift anywhere in the table: a profile either conforms completely or is not JSON.
This is the D-08 finding reproduced by the mechanism that will keep watching it.

### Decision: the check logs, it does not block

It never raises and never stops the merge.

### Reasoning

At 2,766/2,767, refusing to merge on a bad path would discard a whole run's worth of good profiles
over a single leaked reasoning trace, and the profiles are the expensive part. The failure this
track is repairing was never that bad profiles got merged — it was that **five features read zero
for the life of the model and nothing anywhere said so**. Detection is the missing capability;
enforcement is not. If a future reading shows drift at scale, the log line is the signal to revisit
this, and the decision is cheap to reverse.

One consequence to accept knowingly: a WARNING in Cloud Run logs is only useful if someone reads it.
No alerting is being added in this track.

### Note on the query's default

`sql_conformance_check` does **not** filter NULL source values by default, because in the scratch
table a NULL `AI.GENERATE` result is a *failed* profile and belongs in the unparseable count.
Auditing `fsa_master` needs the explicit predicate above, or all 8,501 never-profiled rows land in
that same count and the reading is meaningless. Both behaviours are pinned by tests.

---

## D-13 — The migration departs from the script the plan told it to copy

**Date:** 2026-09-23 · **Phase:** 4 · **Status:** decided, executed

### What ran

17 nullable columns added to `fsa_master` and `fsa_master_backup_20260923`, generated from
`PILLAR_FIELDS` + `NON_JSON_COLUMNS`. Approved as "snapshot first, then prod". Cost £0 —
`ADD COLUMN` is metadata-only.

| | rows | labels | fingerprint |
|---|---|---|---|
| `fsa_master` before | 11,268 | 411 | 7075448033881697774 |
| `fsa_master` after | 11,268 | 411 | 7075448033881697774 |
| after a second idempotent run | 11,268 | 411 | 7075448033881697774 |
| `fsa_master_backup_20260923` before | 11,268 | 411 | 7075448033881697774 |
| `fsa_master_backup_20260923` after | 11,268 | 411 | 7075448033881697774 |

The snapshot's pre-migration hash equals production's, which is a stronger statement than the row
and label counts recorded in Phase 0: the restore point is byte-for-byte production, not merely the
same shape.

### Three departures from `migrate_to_in_scope_workflow.py`

The plan said to follow that script. Three of its habits are wrong for this table.

**Dry run is the default.** It executes unless told otherwise. This one requires `--execute`. The
table holds the only copy of 411 hand-entered labels and there is no non-production environment;
the safe default is the one where forgetting a flag costs nothing.

**A failed `ALTER TABLE` raises.** It downgrades DDL failures to `logger.warning` and continues, so
a partial schema looks like a successful run. Phase 5's backfill targets these columns by name; a
half-applied schema would surface there instead, further from the cause.

**The dry run submits rather than prints.** It prints the SQL. Printing SQL nobody parsed is not
validation. Submitting each statement with `dry_run=True` earned its keep on the first run:
`COUNT(*) AS rows` is a syntax error, `ROWS` being reserved for window frames. That would have
failed mid-execute, against production, after the approval.

### The fingerprint had to be subtractive, not snapshotted

`PRE_EXISTING_COLUMNS` was first derived from `MASTER_BQ_SCHEMA`. Updating that constant to list the
new columns — the very next step — silently changed what the fingerprint covered. Two existing tests
failed, which is the only reason it was noticed.

The consequence would not have shown up here (all 17 columns are NULL, so the hash is unchanged
either way). It would have shown up in Phase 5: once the backfill populates them, a re-run of the
migration would compute a legitimately different hash, conclude the data had been corrupted, and
refuse to proceed. `PRE_EXISTING_COLUMNS` now subtracts `NEW_COLUMNS` from the load schema, which is
stable across both edits, and a test asserts the subtraction leaves 27 columns — a fingerprint over
no columns passes every comparison it is given.

### Ordering, held

The DDL ran before `MASTER_BQ_SCHEMA` was updated, and Phase 4 stayed off `main` until both were
done. That constant is the load schema for the weekly cron; `load_table_from_json` fails the entire
load if it names a column the table lacks, and `main` auto-deploys. The reverse order is safe — a
table with columns the load schema omits just gets NULLs — so the risk is one-directional and the
sequence is not optional. Verified live afterwards: 44 declared, 44 present, none missing.

Nothing in the repo reconciles the two automatically. The 17 fields are now *appended from* the
canonical schema rather than retyped, so the failure mode that remains is adding a `PillarField`
without running the migration — which the module docstring calls out, and which a future phase could
close with a startup assertion if it proves to be a real risk.

---

## D-14 — Phase 5: the `in_scope` re-derivation the plan specified would have destroyed 58% of the training set

*2026-09-23*

The backfill itself was uneventful: five `UPDATE`s over data the table already held, 2,767 / 2,767 /
2,263 / 243 / 1,263 rows, no model calls, £0. Every typed column came out at 2,766 filled — the
2,767th is the known unparseable reasoning-trace row, FHRSID 1855447 — and every feature column has
more than one distinct value, which is the condition this track exists to create. A cross-check
against a fresh re-read of the raw JSON found 0 mismatches across all 2,766 rows on an INT64, a
STRING-enum and a BOOL representative, so the extraction is not merely plausible but exact.

The interesting part is `in_scope`.

**What the plan said.** Re-derive `in_scope` from `pillar_is_sit_down` for the 1,476 rows that
disagree with their own profile, since `migrate_to_in_scope_workflow.py` gated on a path
(convention B) that has never resolved and so assigned scope from `maps_types` alone.

**What measurement found.** Re-measured against the backfilled column the disagreement is 1,479
(1,458 leaving scope, 18 entering, 3 filling a NULL). But **216 of those rows carry a human
`user_rating` or `rating_source`**, and **214 of the 369 trainable labels sit on rows the profiler
calls not-sit-down**. The blind re-derivation would have taken the training set from 369 to 157.

Worse than the count: the rows it removes have a mean rating of 1.5 against 2.55 overall, and **not
one of them is rated 6 or above**. So the profiler is not wrong about them — they really are
takeaways and cafés. The problem is that `in_scope` is doing two incompatible jobs. As a spend gate
it should follow the profiler. As a training-set filter it would be discarding exactly the negative
examples that teach the model what a bad match looks like, on rows a human already paid attention
to. Re-deriving a human's decision from a model's opinion is the wrong direction of authority
regardless of which one is right.

**Decision.** The re-derivation skips any row with a `user_rating` or `rating_source`. 1,263 rows
corrected, 216 human decisions preserved, training set unchanged at 369, triage queue 10,841 →
9,610. The guard gives up 216 of 1,479 corrections, which is the cheapest part of the change. Both
options were measured and put to the user before anything ran; the guarded version was chosen.

The deeper problem — one column serving as both spend gate and training filter — is not fixed here.
Phase 9 has the evidence to decide whether the training `SELECT` should stop filtering on `in_scope`
at all, at which point the 42 currently-excluded labels come back too.

### Three other things Phase 5 turned up

**The sentinel guard exists in three places, not one.** The plan named `enrich_maps_data.py:26`.
`ml_prediction.py:41` and `train_bqml_model.py:88` make the same decision with the same hand-copied
predicate. Nulling 243 sentinel ratings without moving all three would have made every prediction
*and every training run* re-query the paid Places API for permanent misses — D1 again, in the Maps
dimension, on the one path that runs on a schedule. Pulled the guard move forward from Phase 6 and
shipped it immediately after the production backfill, with the user's go-ahead, because the gap
between the two is the exposure window. The `train_bqml_model.py` pre-flight also had to change what
it *selects*: the new guard would otherwise have raised `AttributeError` into a bare `except` that
downgrades the failure to a warning and silently skips all enrichment.

The **writer** had to change with them. A Places miss now records `maps_found = FALSE` and a lookup
timestamp instead of a fresh `-1`; otherwise the cleanup is a one-way door and the next miss puts
itself straight back in the queue. Two intended side effects fall out: the 243 miss rows move from a
zero Maps quality prior to the neutral 50 in `calculate_restaurant_priority` — they were being
ranked below restaurants nobody had ever looked up — and the UI's "Has Google Maps Rating" filter
stops counting a `-1` as a rating. Both were Phase 8 items that the data change delivered for free.

**Rehearsing on the snapshot would have destroyed the snapshot.** The plan says "run on the snapshot
copy first, compare, then prod", and separately designates `fsa_master_backup_20260923` as the
restore point. Those are incompatible: a backfilled snapshot restores backfilled data. Rehearsed on
a throwaway `fsa_master_rehearsal_20260923` with a 7-day expiry instead, left the snapshot pristine,
and then verified production byte-identical to the validated rehearsal across all 19 affected
columns via `FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(...)))` per row. Phase 4's fingerprint proved
*nothing changed*; this one proves *the right thing changed*.

**The `in_scope` preview reads 0 in a pure dry run**, because its count query shares a predicate with
its `UPDATE` and that predicate reads a column an earlier statement in the same run fills. Keeping
the shared predicate was deliberate. The alternative — computing the preview from the JSON while the
`UPDATE` reads the column — is an estimate derived differently from the behaviour it predicts, which
is precisely the shape of D1. The rehearsal run is what makes the number visible before production,
and it reported 1,263, matching the independent measurement exactly.


---

## Measurements

*Populated by Phase 0 recon, 2026-09-23.*

| Measurement | Value | Date |
|---|---|---|
| Snapshot table | `fsa_master_backup_20260923` | 2026-09-23 |
| Snapshot row count | **11,268** (matches production) | 2026-09-23 |
| Snapshot `COUNT(user_rating)` | **411** (matches production) | 2026-09-23 |
| Observed top-level key census | convention A — 6 numeric-prefixed pillar objects + `match_score` + `summary_reasoning` | 2026-09-23 |
| Shape stable across rows? | **Yes** — 2,766/2,767 identical; 1 unparseable payload | 2026-09-23 |
| Rows profiled (`gemini_insights_structured`) | 2,767 | 2026-09-23 |
| `COUNT(*) WHERE gemini_insights IS NOT NULL` (expected 0) | **1,116** — disjoint from V2 | 2026-09-23 |
| Flat pillar paths resolving (D2) | **0 of 2,766** for all five; `match_score` 2,766 | 2026-09-23 |
| Nested pillar paths resolving | 2,766 of 2,766 for all 17 | 2026-09-23 |
| Rows with `maps_rating = -1` | 243 | 2026-09-23 |
| Rows with a Maps hit | 2,263 | 2026-09-23 |
| Rows never looked up in Maps | 8,762 | 2026-09-23 |
| Rows with lat/lon | 2,291 | 2026-09-23 |
| `in_scope` true / false / null | 10,693 / 427 / 148 | 2026-09-23 |
| `in_scope` rows contradicting `pillar_is_sit_down` (D13) | **1,476** (1,458 + 18) | 2026-09-23 |
| D13 disagreements re-measured on the typed column | **1,479** (1,458 out / 18 in / 3 NULL) | 2026-09-23 |
| D13 disagreements carrying a human decision | **216** — protected by the guard | 2026-09-23 |
| Trainable labels under a *blind* re-derivation | **157** of 369 — rejected | 2026-09-23 |
| Phase 5 rows updated (5 statements) | 2,767 / 2,767 / 2,263 / 243 / 1,263 | 2026-09-23 |
| Pillar columns filled after backfill | **2,766 of 2,767** on all 14 | 2026-09-23 |
| Distinct values per feature column | match_score 91, value 20, community 22, linguistic 17, culinary 23, geo_specificity 3, is_sit_down 2, establishment_type 3 | 2026-09-23 |
| Backfill vs. fresh JSON re-read | **0 mismatches** over 2,766 rows (INT64, STRING-enum, BOOL) | 2026-09-23 |
| Production vs. validated rehearsal, 19 affected columns | **0 rows differ** | 2026-09-23 |
| Surviving `maps_rating = -1` sentinels | **0** | 2026-09-23 |
| `in_scope` true, before → after Phase 5 | 10,693 → **9,465** | 2026-09-23 |
| Labels / trainable, before → after Phase 5 | 411 → 411, 369 → **369** | 2026-09-23 |
| Rows with a prediction | 1,065 | 2026-09-23 |
| Labelled rows / labelled **and** profiled | 411 / **404** | 2026-09-23 |
| Labelled rows passing the training `in_scope` filter | **369** (42 excluded) | 2026-09-23 |
| Phase 2 split (`FARM_FINGERPRINT(fhrsid) MOD 5`) | 292 train / 77 holdout | 2026-09-23 |
| Split mean rating, train / holdout | 2.613 / 2.299 | 2026-09-23 |
| Baseline model MAE / RMSE / R² / Spearman | **0.847 / 1.283 / 0.545 / 0.545** | 2026-09-23 |
| `match_score`-only baseline MAE / RMSE / R² / Spearman | **0.697 / 1.095 / 0.668 / 0.603** | 2026-09-23 |
| Training-mean floor MAE / RMSE | 1.566 / 1.926 | 2026-09-23 |
| Conformance check, first production reading | **2,766 / 2,767** conform on all 14 paths; 1 unparseable; 0 partial | 2026-09-23 |
| `fsa_master` data fingerprint (27 pre-existing columns) | `rows=11268 labels=411 hash=7075448033881697774` | 2026-09-23 |
| Snapshot fingerprint, pre-migration | **identical to production** — byte-for-byte restore point | 2026-09-23 |
| Columns after Phase 4 (both tables) | **44** (27 + 17), all nullable, types verified | 2026-09-23 |

## Cost ledger

| Item | Basis | Estimate |
|---|---|---|
| Phase 0 recon (8 read-only queries) | 33.4 MiB scanned @ $6.25/TiB | **£0.0002** — spent |
| Phase 3 conformance reading | 4.5 MiB scanned @ $6.25/TiB | **£0.00003** — spent |
| Phase 3 conformance check, per enrichment run | one scan of the scratch table, ~KiB | **£0** in practice |
| Phase 0 snapshot | 11,268 rows ≈ 6 MiB active storage | **< £0.01/month** — spent |
| Phase 4 `ALTER TABLE ADD COLUMN` × 17 | metadata-only | **£0** — spent, both tables |
| Phase 5 backfill (pure SQL over existing JSON) | 5 `UPDATE`s, ~6 MiB each | **< £0.01** — spent |
| Phase 5 `in_scope` re-derivation | one `UPDATE` over 1,263 rows | **< £0.01** — spent |
| Phase 5 rehearsal copy | 11,268 rows, 7-day expiry | **< £0.01** — spent |
| Places re-query of 243 permanent misses | avoided by the guard move | **~£6 avoided, recurring** |
| Phase 9 retrain | `BOOSTED_TREE_REGRESSOR` over 404 rows | **~£0.05** |
| **Legacy re-profile — withdrawn** | would have been 2,767 × `AI.GENERATE` | **avoided** |
| Orphaned scratch tables dropped | `recents`, `genairesults_temp`, 2 × `temp_update_reviews_*` | **£0** — done, 2026-09-23 |
| Phase 7 sweep, tokens | 1,116 × `gemini-3.8-flash` @ $0.75/$3.75 per 1M | **$7–$24** (£6–£19) |
| Phase 7 sweep, grounding | 1,116+ searches; 5,000/month free across Gemini 3.x, then $14/1,000 | **$0–$35** (£0–£28) |
| Phase 7 sweep, Places for the 975 without Maps | incurred anyway on first prediction, not by the sweep | **~$31** (£25), separate |
| **Phase 7 sweep, total** | | **$7–$59 / £6–£47, pending a measured pilot** |

The only line item that needs its own approval is the Phase 7 sweep; everything up to Phase 9 is
pennies because the backfill turned out to be pure SQL.

The Phase 7 range is wide for two reasons that no amount of arithmetic will close: grounded calls
inject retrieved search results into the input, and `gemini-3.8-flash` bills thinking tokens as
output. Both are unobservable from the stored `.result`. `AI.GENERATE` returns `usageMetadata` in
its `full_response` field, which the current script discards -- a 20-row pilot costing roughly $0.15
would replace the range with a measured per-call figure before anything is committed.
