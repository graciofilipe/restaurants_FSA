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

## D-15 — Phase 6 cannot be merged on its own: the fix changes the model's input schema

**Date:** 2026-09-23 · **Phase:** 6 · **Status:** decided, retrain pending approval

The plan's merge rule is that every phase boundary is green *and independently deployable*, because
`main` auto-deploys to Cloud Run. Phase 6 is the first phase where that does not hold, and the
reason is worth recording because it is a property of the repair, not an oversight.

Fixing D2 means the `ML.PREDICT` subquery stops emitting `score_1_value_and_volume_rating` and
starts emitting `pillar_value_rating`. `ML.FEATURE_INFO` on the live
`restaurant_preference_model` lists **19 input features** under the old names. BQML requires the
prediction input to carry every column the model was trained on, so the moment this lands,
"Generate Predictions" fails outright.

Worth being precise that **failing is the good outcome here**. Had the aliases been left unchanged
and only the paths fixed, the model would have kept predicting — from features it was trained to
see as constant 0 and would now receive as real values between 1 and 10. That is silent train/serve
skew producing plausible-looking numbers, and it is exactly the failure mode the D3 parity test was
written to prevent. The alias change converts it into a loud one.

So the retrain Phase 9 schedules has to happen **before** Phase 6 merges, not after. Phase 9 keeps
the *evaluation* — the harness re-run, the A-vs-C `in_scope` comparison, the keep-or-retire verdict.
Only the `CREATE OR REPLACE MODEL` moves forward. This also matches the stated cost principle:
training is cheap, and it is the Gemini and Places generation that is not.

Measured against production before committing to it — training population 370 rows; the JIT
pre-flight would fire **0 Places lookups, 0 postcode lookups, 7 Gemini calls**, those 7 being
labelled rows that have never been profiled. ≈ $0.02 in tokens, grounding inside the free monthly
allowance, BQML training over a 5.7 MB scan. **Under £0.05.** 363 of the 370 already carry every
typed pillar column, so the retrain is reading real features on day one rather than waiting for a
sweep.

There is an unavoidable few-minute window in either order — retrain first and the deployed old code
sends old aliases to a new-schema model; merge first and the new code hits the old model. Retraining
first is the better half: the failure is confined to a button the user presses by hand, not to the
weekly cron.

---

## D-16 — A NULL postcode silently voided the profile prompt (new defect)

**Date:** 2026-09-23 · **Phase:** 6 · **Status:** fixed in code (e6ad77c); data repair pending

Found by running the retrain, not by reading the code. The JIT pre-flight reported 7 labelled rows
missing a profile; all 7 came back unparseable, and all 7 have `postcode IS NULL`.

`SCRIPT_GENERATE_INSIGHTS` concatenated `postcode` straight into the prompt while `COALESCE`-ing the
three address lines on either side of it. Concatenating NULL in BigQuery yields NULL, so the whole
prompt was NULL and `AI.GENERATE` returned nothing — no error, no cost signal, just a row that stays
unprofiled. Every prediction run and every training run then retried it, because the JIT guard reads
`gemini_insights_structured IS NULL` and that is exactly what a failed generation leaves behind.

**126 unprofiled rows have a NULL postcode**, 7 of them labelled (two rated 5 and 7 — among the more
informative labels the model has). They are not a random 126: `enrich_maps_data.py` searches Places
by `BusinessName + PostCode`, so a missing postcode degrades that lookup too. The inspected names
are chains and concessions — `Costa Coffee Drive Thru`, `UNIT R13 VICTORIA PLACE`, `Dub Pan` — the
FSA rows least likely to carry a clean postcode.

`businessname` had the identical exposure and now gets the identical guard, though nothing currently
violates it.

**A defect the dual-write introduced, in the same run.** The merge stamped
`gemini_profiled_at = CURRENT_TIMESTAMP()` unconditionally on match, so those 7 rows are now marked
profiled while holding no profile. That state is worse than either end of it: a staleness sweep
reads them as fresh and skips them, while the JIT guard reads them as missing and retries them. The
merge now carries `WHEN MATCHED AND S.gemini_insights IS NOT NULL`, so a failed generation leaves
the row untouched. The 7 already-stamped rows need a one-line `UPDATE` to clear.

**The Phase 3 conformance check is what surfaced this**, logging
`7 generated, non-conforming paths -- ... unparseable=7` at the moment it happened. It was built to
detect schema drift and caught a prompt bug instead, which is the argument for having built it.
D-12 recorded the decision to keep it advisory rather than blocking; had it been blocking, this run
would have aborted before training and the finding would have looked like an outage.

---

## D-17 — Staleness was switched on while nothing is stale, deliberately

**Date:** 2026-09-23 · **Phase:** 7 · **Status:** decided

`GEMINI_PROFILE_MAX_AGE_DAYS = 180`, checked against `gemini_profiled_at`, now decides whether the
Predict button re-profiles a restaurant. Measured before choosing it: 2,767 profiled rows, every one
stamped, **0 stale** — the oldest stamp is the Phase 5 backfill from earlier today, so the first row
becomes eligible on 2027-03-22.

That is the point. Phase 5 chose `CURRENT_TIMESTAMP()` over NULL for exactly this reason, and it
means the mechanism can land, be tested and be reverted six months before it can spend anything. The
alternative — ship the predicate later, when rows are genuinely old — would have put the code change
and its first bill on the same day.

Three decisions inside it, each of which could have gone the expensive way:

- **Unknown age reads as fresh.** A row with a profile but no timestamp is not re-profiled. There
  are zero such rows, and if the backfill had missed some, treating a missing timestamp as "old"
  would re-profile them all on the next click. A parse failure on the timestamp is handled the same
  way.
- **The training pre-flight passes `max_age_days=None`.** It fills gaps and never refreshes.
  Retraining is cheap, re-profiling is not, and training is the only scheduled caller — a staleness
  rule there spends money with nobody watching. This is the user's own framing, recorded as a
  parameter rather than a comment.
- **180 days, not 30 or 90.** A profile describes cuisine, menu language and neighbourhood. Halving
  the threshold doubles the recurring bill for signal that moves on the scale of a refurbishment.

**The D1 half.** The UI's "Estimated New Gemini Calls" and the enrichment it estimates were two
implementations of one question. Both now call `needs_gemini_profile`. The estimate also answers
correctly when "Force Regenerate" is ticked, which it previously ignored — it would show 0 and then
bill for the whole batch.

> **Corrected 2026-09-23 (Phase 8).** This entry originally said the two surfaces "agreed only
> because the legacy column they disagreed about is NULL on every row." That is wrong on both
> counts. `gemini_insights` is non-NULL on **1,116 rows** — V1 free text, written before the V2
> merge started nulling the column, on rows that have *no* structured profile. So the two surfaces
> disagreed on exactly those 1,116: the estimate counted a row as cached if **either** column held
> anything, the executor asked only about the structured one. Put a V1-text row in a batch and the
> estimate said 0 calls while the run billed for one. The Phase 7 fix is therefore load-bearing
> rather than tidy-up, and the direction of the old error was always to under-report the bill. See
> [D-18], which is the same NULL, mishandled a second way.

---

## D-18 — NaN is truthy, so `a or b` scored 10,152 rows as never profiled (new defect)

**Date:** 2026-09-23 · **Phase:** 8 · **Status:** fixed in `0d02c8c`

Found while measuring whether Phase 8's staleness switch changed any ordering. It changed 1,051
rows, which it should not have — the two expressions were supposed to be equivalent. They were not,
and the reason is a Python detail with a five-figure blast radius:

```python
gemini_val = row.get('gemini_insights') or row.get('gemini_insights_structured')
```

A BigQuery NULL arrives in a DataFrame as `float('nan')`, and **NaN is truthy**. So for every row
whose V1 text is NULL — 10,152 of 11,268 — the `or` stopped at the first operand and returned NaN.
The very next line asks `pd.isna(gemini_val)`, gets True, and awards the maximum staleness score of
100, which is the "never scored, profile it" bucket.

**1,020 of those rows hold a structured profile *and* a prediction.** They sat at the top of the
Budget Allocator's queue being offered for re-profiling, indistinguishable from rows that had never
been touched. The heuristic that exists to direct Gemini spend at the rows that need it was
directing it at the rows that needed it least.

The mirror-image error rode along: the 1,116 rows that hold **only** V1 text read as profiled, when
what they have is a text blob from the superseded profiler and no V2 columns at all.

**Fix.** Staleness reads `gemini_profiled_at` — one typed column, written by the V2 merge, measured
as exactly co-extensive with `gemini_insights_structured` (2,767 each, 0 rows either way), and one
that survives the Phase 10 drop of the legacy column. `first_present()` now exists for "first
non-missing of these keys" and the postcode read uses it, because the same trap was laid there:
`row.get('postcode') or row.get('PostCode')` returned NaN, `str(nan)` is `'nan'`, and the centroid
lookup resolved that to central London.

**Why the tests did not catch it.** Every fixture in `test_scoring_priority.py` set
`gemini_insights` and `gemini_insights_structured` explicitly, to `None` or to a string. `None or x`
returns `x`; only `nan or x` returns `nan`. The frames the tests built could not reproduce the
frames BigQuery produces. The two new tests pass NaN deliberately.

**Related.** [D-01] is the same conflation of the two columns in the JIT guard; [D-17] records the
corrected version of what `gemini_insights` actually contains.

---

## D-19 — Verdict: the repair worked, and the model still has not earned its place

*Measured 2026-09-23 by `scripts/evaluate_model.py --execute`, unchanged from Phase 2 and with the
same `--holdout_modulus 5`. Same split, same 292/77 rows, same code.*

### The four numbers

| Predictor | Features | MAE | RMSE | R² | Spearman | median AE |
|---|---|---|---|---|---|---|
| Boosted tree, **Phase 2** (5 pillars dead) | 19 | 0.847 | 1.283 | 0.545 | 0.545 | — |
| Boosted tree, **Phase 9** (repaired) | 21 | **0.737** | 1.320 | 0.518 | **0.689** | **0.308** |
| `match_score` alone (`LINEAR_REG`) | 1 | **0.679** | **1.028** | **0.707** | 0.645 | 0.492 |
| Training mean | 0 | 1.566 | 1.926 | 0.0 | n/a | — |

**What the repair bought:** MAE 0.847 → 0.737 (−13%) and Spearman 0.545 → 0.689 (+0.144). On the
typical row the repaired tree is now the most accurate predictor in the table — a median absolute
error of 0.31 against `match_score`'s 0.49. The five pillars that were pinned to zero are carrying
signal, which is the question Phase 2 existed to make answerable.

**What it did not buy:** the one-feature baseline still wins on MAE, RMSE and R². The tree's error
distribution is the story — same typical row, fatter tail:

| | tree | match_score |
|---|---|---|
| median abs. error | **0.308** | 0.492 |
| 90th-percentile abs. error | 2.101 | **1.251** |
| worst error | 5.226 | **4.749** |
| rows off by more than 2 | 8 / 77 | **5 / 77** |

### The comparison is not significant, and saying so is the point

Paired bootstrap over the 77 held-out rows, 20,000 resamples:

| Difference | Point | 95% CI | P(tree better) |
|---|---|---|---|
| MAE(tree) − MAE(match) | +0.058 | [−0.073, +0.210] | 0.21 |
| ρ(tree) − ρ(match) | +0.044 | [−0.055, +0.156] | 0.79 |

Both intervals straddle zero. With 77 rows the honest statement is **"no detectable difference"**,
not "the baseline wins" and not "the tree caught up". Phase 2's headline — *beaten on every metric*
— no longer holds; nothing has replaced it.

### The part of the ranking the app actually reads

Spearman scores the whole list. The user reads the top of it. Mean true rating of each predictor's
top-k, against an oracle that ranks by the label itself (overall mean 2.30):

| k | tree | `match_score` | oracle |
|---|---|---|---|
| 5 | 5.60 | **6.60** | 7.00 |
| 10 | 5.20 | **6.10** | 6.70 |
| 20 | 4.45 | 4.40 | 4.90 |

At the head of the queue `match_score` still picks better, and it is close to the oracle. This is
the one place the two predictors visibly disagree, and it is the place that matters.

### Recommendation: keep BQML, do not trust it above `match_score` yet

Phase 2 pre-registered three outcomes. The result is outcome 2 — *improves but still trails* — but
the pre-registered response to it ("ship the linear baseline and retire BQML") was written before
the ranking metric flipped, and acting on it now would be reading a null result as a verdict.

1. **Keep the model.** It costs pennies to train, and the repair moved it a long way in one step.
   Retiring it would discard the only surface that can learn from the labels being collected.
2. **Do not promote it over `match_score` at the top of the queue.** On top-5 and top-10 picks the
   single Gemini number is still better, and that is where a wrong ordering costs a real visit.
3. **Re-measure at ~600 in-scope labels** (369 today) with this same harness and modulus. That is
   the cheapest possible tiebreaker, and at 77 holdout rows nothing smaller will settle it.
4. **If it still has not separated at 600 labels**, retire BQML in a separate track and rank on
   `match_score`. The 21-feature tree would then be paying maintenance — train/serve parity, schema
   coupling, retrain-on-feature-change — for signal a single JSON field already carries.

The evidence for that decision is now on disk and reproducible with one command. Retiring BQML
remains out of scope for this track, as the plan states.

### Two caveats, recorded rather than buried

The `match_score` baseline also moved — 0.697 → 0.679 MAE — although the harness is byte-identical.
The data underneath it changed: Phase 5's `in_scope` re-derivation and the 7 profiles the Phase 6
retrain filled in altered which rows carry a `match_score` and what it is. The split is unchanged
(292/77, identical means), so the comparison within this run is sound; the cross-phase comparison of
the *baseline* to itself is approximate. The tree's 0.847 → 0.737 spans the same data change.

The production model was not retrained in this phase. Phase 6 already retrained it on the corrected
features, and its 21 inputs were verified against `feature_select_list()` here — exact match, label
excluded. Phase 8 changed no features, so a second retrain would have produced the same model.

**Related.** [D-11] is the Phase 2 floor this replaces; [D-15] is the retrain whose features are
being measured; [D-20] is the stale-prediction sweep this verdict makes necessary.

---

## D-20 — Every prediction in the table was made by a model that no longer exists

*Measured 2026-09-23 by `scripts/invalidate_stale_predictions.py`.*

All **1,065** predictions predate the Phase 6 retrain — `current_predictions` is 0. They came out of
a 19-feature model with five constant-zero pillars, and the model that would score those rows today
is a different one.

Nothing in the table records this. `predicted_at` says *when* a row was scored, not *what* scored
it, and the staleness component reads a recent timestamp as "freshly scored" and parks the row at
the bottom of the queue. A prediction from a retired model is therefore self-perpetuating: it is
wrong, and it is the reason the row never comes back up to be re-scored.

**Fix.** `scripts/invalidate_stale_predictions.py` nulls `predicted_user_rating` and `predicted_at`
where `predicted_at < ` the served model's training time, read from the model's own metadata rather
than typed in. `created` rather than `modified`: `CREATE OR REPLACE MODEL` resets creation time, so
for a BQML model it is the training time, while `modified` also moves for a description or label
edit — which would silently widen the cutoff and clear good rows.

**Cost of the consequence, not of the write.** The `UPDATE` is 11.2 MiB. What follows it is not
free: 1,022 of the 1,065 rows already hold a Gemini profile and re-score for the price of
`ML.PREDICT`, but **43 do not**, and those bill a fresh `AI.GENERATE` on the way through — about
£0.30 at the Phase 6 measured rate. The dry run reports the two counts separately so that number is
visible before the write rather than after it. 32 of the cleared rows carry a human `user_rating`;
they lose a displayed prediction from a retired model and are recoverable from
`fsa_master_backup_20260923`.

**Executed 2026-09-23 with explicit user approval: 1,065 cleared**, row count and label count
unchanged. The first `--execute` attempt was refused by the sandbox's auto-mode classifier, which
read the bulk `UPDATE` as a mass delete; nothing was written in the interim.

### What the empty column does to the queue, until it is refilled

With no predictions anywhere, staleness is **constant 100 on all 11,268 rows**. The component still
computes; it just no longer discriminates, so the queue falls back to proximity, the Maps quality
prior and scope confidence. That temporarily *inverts* the Phase 8 result — unprofiled rows in the
top 25 go 4 → 0 — because the Maps prior now breaks every tie and a row nobody has looked up on
Maps has no prior to offer. Nothing is wrong with either number; they are answers to different
questions, and the second one is what "nothing here has been scored" actually looks like.

The gradient comes back as soon as rows are re-scored. 1,022 of the cleared rows already hold a
Gemini profile and a Maps lookup, so re-scoring them runs `ML.PREDICT` and nothing else.

**Related.** [D-19] is the verdict that makes the sweep necessary; [D-15] is the retrain that made
every one of these predictions stale.

---

## D-21 — The V1 text was the only copy of 1,116 evaluations, so it was archived before it was dropped

**Decision.** Copy `gemini_insights` to `gemini_insights_v1_archive_20260923`, verify the copy, then
`ALTER TABLE ... DROP COLUMN`. Gate the drop on the archive in code, not on remembering to do it.

The column is the free-text output of the profiler that preceded the structured V2 JSON. Phase 0
expected it to be empty — D1 was written as "the JIT guard tests a column that is NULL on every
row" — and found 1,116 populated rows instead, which is why the defect was worse than the plan
assumed. Retiring it therefore meant deciding what those 1,116 rows are worth, not just deleting a
dead column.

### What is in it

| | |
|---|---|
| Rows with text | 1,116 |
| Of those, in scope | 1,090 |
| Of those, ever given a `user_rating` | **0** |
| Of those, also holding a V2 `gemini_insights_structured` | **0** |
| Distinct values | 1,116 — no duplicates |
| Length | min 126, median 1,379, max 3,421 characters |
| `first_seen` range | 2025-06-13 → 2026-01-10 |

The zero that decides it is the fourth row. The V1 and V2 profile sets are **exactly disjoint**:
every row that has V1 text has no structured profile, and every profiled row has no V1 text. So the
column is not a stale duplicate of data held better elsewhere — for those 1,116 restaurants it is
the only judgement anything has ever formed. Nothing reads it (the JIT guard moved to
`gemini_insights_structured` in Phase 1, the model never used it, the UI stopped displaying it),
and the V2 merge nulled it on write, so it was also never going to grow.

Quality is uneven, which argues for keeping it cheaply rather than either trusting it or binning it.
Three real examples: a PASS/FAIL rubric for Firedough Pizza; a correct and non-obvious "this is a
reggaeton club, not a restaurant" for La Vuelta, derived from a TikTok hashtag; and a bare menu
description for BaxterStorey that says nothing evaluative at all.

### Verification, because "the copy worked" is not evidence

| | source | archive |
|---|---|---|
| Rows | 1,116 | 1,116 |
| `BIT_XOR(FARM_FINGERPRINT(gemini_insights))` | −5710637097229171032 | −5710637097229171032 |
| `SUM(LENGTH(gemini_insights))` | 1,526,413 | 1,526,413 |
| Rows in source but not archive | 0 | |

`BIT_XOR`, not `SUM`: summing `FARM_FINGERPRINT` over ~1k rows overflows INT64 and BigQuery raises
rather than wrapping, which is how that was found.

The archive is created with a plain `CREATE TABLE` — not `OR REPLACE`, not `IF NOT EXISTS`. After
the drop, `build_archive_sql` would select from a column that no longer exists; `OR REPLACE` would
let a careless re-run replace a good archive with an error or an empty table, and `IF NOT EXISTS`
would make that same re-run print success. Failing loudly is the correct behaviour for a script
whose whole job is to not lose data.

### Ordering, which is the part that could have broken production

`MASTER_BQ_SCHEMA` is the load schema handed to `append_to_bigquery` by the weekly Cloud Run Job. A
column named there that the table does not have fails the load. So the sequence was: land the code
that stops naming the column → merge 6917678 → build fb2fec3a SUCCESS → revision
`restaurants-fsa-00228-lp4` serving → *then* drop. Re-checked after the drop: `MASTER_BQ_SCHEMA` and
the live table agree on all 43 names, with nothing named-but-absent.

The reverse order would not have failed anything visible. It would have failed the next ingest,
once, a week later, in a job nobody watches.

**What was removed.** `MASTER_BQ_SCHEMA`; `ORIGINAL_COLUMNS_TO_KEEP` (a flat key copy of the FSA
payload — the API has never returned this field, so the entry only reserved a NULL); the
`gemini_insights_status` filter parameter and its branch, which no caller passed; and
`T.gemini_insights = NULL` in `SCRIPT_MERGE_INSIGHTS`, the assignment that did the nulling.

**What deliberately stayed.** `S.gemini_insights` in the same MERGE, and the default `column=` of
`sql_conformance_check`. Both name the *scratch* table's alias for the raw `AI.GENERATE` output, not
the master column. Same string, different object; now commented as such.

**State after.** 44 → 43 columns. 11,268 rows and 411 labels unchanged. The Phase 0 snapshot
`fsa_master_backup_20260923` still exists and still carries the column, so the drop is reversible
from two places until that snapshot is deleted in Phase 12 — at which point the archive table is the
only copy, and it is not scheduled for deletion.

**Not decided here.** Whether those 1,090 in-scope restaurants are worth re-profiling into V2. That
is the Phase 7 sweep, £6–£47, still pending a 20-row pilot and its own go-ahead. Dropping the column
does not foreclose it: the archive holds the text, and a V2 profile would not have read it anyway.

**Related.** [D-08] is where the 1,116 were found; [D-01] is the guard that used to read this column.

---

## D-22 — `manual_review` says `rejected` about 9,348 restaurants that are in scope, so it was replaced rather than archived

**Date:** 2026-09-23
**Phase:** 10

**Context.** `manual_review` is a free-text status column predating `in_scope` and `rating_source`.
The plan called for replacing, not deleting, the one place it still had teeth: the default filter in
`execute_gemini_enrichment`, `manual_review IN ('pending', 'not reviewed')`, which decides who gets
paid for a Gemini profile.

**What the column actually contains,** measured before anything was changed:

| Value | Rows | of which `in_scope = TRUE` | of which labelled |
|---|---|---|---|
| `rejected` | 10,869 | **9,348** | **351** |
| NULL | 288 | — | — |
| `pending` | 109 | — | — |
| `prending` | 2 | — | — |

So the dominant value is a no about restaurants the user has separately judged to be in scope, and
351 of them carry a hand-entered rating — a rejection of things that were not rejected. Add a
`prending` typo that the `IN (...)` predicate silently excluded from enrichment on two rows. The
column is not a record of decisions; it is the residue of a workflow that `in_scope` replaced.

**The contrast with [D-21] is the whole decision.** The V1 text was 1,116 distinct Gemini
evaluations, disjoint from every V2 profile, and dropping it would have destroyed the only copy — so
it was archived first, at real effort. `manual_review` is one of four values, none of which means
what it says, all of which are recoverable from the snapshot for as long as the snapshot exists. An
archive table would be a ceremony performed on noise. **No archive is proposed.**

**The replacement predicate: `in_scope IS NOT FALSE`, not `IS TRUE`.** Rows arrive untriaged with
`in_scope` NULL, and profiling is usually what answers the question, so `IS TRUE` would mean a new
restaurant is never looked at — the enrichment queue would drain to nothing. What `IS NOT FALSE`
does exclude is the already-answered no: a cafe or a bakery, judged by triage or by an earlier
profile, which there is no reason to pay to profile again.

Measured over the current 33-day window: the old predicate selects **153** rows, the new one **146**.
The seven dropped are all `in_scope = FALSE`. This is strictly fewer `AI.GENERATE` calls, not a
different set — the change cannot cost more than it saves. Dry-run validated against `fsa_master`
before the commit.

**Also removed, and worth naming.** The `review_status_filter` parameter on both
`execute_gemini_enrichment` and `load_filtered_data_from_bq`. No caller has ever passed either. Two
public parameters that existed only to let a caller widen the filter, in a codebase where widening
the filter is what spends money.

**Ordering, again.** Code deployed before the drop, for the reason in [D-21]: `MASTER_BQ_SCHEMA` is
the weekly cron's load schema, and naming a column the table lacks fails the scheduled ingest in a
Cloud Run Job nobody watches. The reverse — the table holding a NULLABLE column the schema omits — is
harmless.

**State after.** Dropped on the go-ahead of 2026-09-23. **43 → 42 columns.** 11,268 rows, 411
labels, 9,465 `in_scope = TRUE` and 2,774 structured profiles all unchanged. `MASTER_BQ_SCHEMA` and
the live table agree on all 42 names in both directions — nothing named-but-absent, nothing
present-but-unnamed. `fsa_master_backup_20260923` still carries the column, so this is reversible
until Phase 12 retires the snapshot; after that it is gone, which is the intended end state.

With `gemini_insights` this makes **44 → 42** across Phase 10.

**Related.** [D-21] is the same retirement for a column that did hold information; [D-05] and [D-13]
are how `in_scope` came to be derived, including the branches that never fired.

---

## D-23 — The error handling that was already there had never run (D9)

**Date.** 2026-09-24. **Phase 11.** Commit `0d9ddc1`.

`load_data_into_state` in `st_app.py` wraps its BigQuery read in `try`/`except` and calls
`st.error` on failure. Two more `try`/`except` blocks guard the sidebar dropdowns. All three were
**unreachable**. `load_filtered_data_from_bq`, `get_distinct_local_authorities` and
`get_distinct_outcodes` each caught their own exception, logged it, and returned `[]` — so the
caller's handler never saw anything to handle, and `[]` is also exactly what "nothing matched your
filters" looks like.

The observable result: an expired credential rendered as **"No data found matching criteria."**
That is worse than an unhelpful message. It is a specific, confident, wrong diagnosis — it tells
the user their filters are too narrow, so the reasonable next action is to widen them, and no
amount of widening reaches a 403.

The three readers now raise `BigQueryExecutionError` with the original chained. The two dropdown
call sites still catch and fall back to an empty list, which is a deliberate asymmetry: the sidebar
renders before the user can press anything, so raising there would take the whole page down and
leave nowhere to read the message. An empty dropdown *next to a visible error* is not claiming the
table is empty.

**The worse half of the same defect, found while fixing it.** `fetch_weekly.main()` — the weekly
Cloud Run Job, the top of the entire pipeline — caught every exception, logged it, and returned.
The process therefore exited 0 and Cloud Run recorded a success. Nobody watches this job, which is
the point of scheduling it; a month of failed ingests would have looked identical to a month of
quiet weeks, evidenced only by a `first_seen` gap nobody would attribute to it. Exit status is the
only signal Cloud Run reads, so it has to be true.

Three distinctions kept, because "raise on everything" would just get the alert muted:

- One bad search area does not cost the others their run. Failures are collected and reported
  together in the exit message, with a count — three of three failing is an outage, one of three is
  a bad config row.
- An **empty** config table stays a warning. Emptying it is how the cron gets paused.
- An unreadable config table fails, and fails first. It is the earliest BigQuery call the job makes
  and therefore where a dead credential surfaces.

`append_to_bigquery` reports failure by returning `False`; the caller logged "Append failed." and
returned normally, so the new restaurants were dropped and the run still counted as a success. It
raises now.

**Related.** [D-18] is the other defect of this family — a wrong answer that looked like an answer,
rather than an error that looked like a result.

---

## D-24 — `np.round` is not `round`, and the priority queue is sorted on the difference (D8)

**Date.** 2026-09-24. **Phase 11.** Commits `762933d` (fixture), `4eab3d8` (refactor).

`calculate_restaurant_priority` walked 11,268 rows with `.iterrows()` and re-sorted the 310-key
outcode dictionary inside the per-row fallback. Measured: **0.80s per full pass**, and the
Streamlit ML Predictions tab re-scores the whole frame on *every rerun* — every slider drag, every
checkbox anywhere on the page — so that was per interaction, not per load.

The interesting part was not the speed-up. It was establishing that the rewrite changed nothing.

**What was not re-derived.** The scoring loop's rules live in `try`/`except` and `isinstance`
chains: outcode resolution, the timestamp coercion that accepts a tz-aware `Timestamp` or an ISO
string or nothing, the `in_scope` ladder that answers to `True` and `1` and `"true"`. Rewriting
those as array expressions means re-deriving them, and a re-derivation can be wrong in ways nothing
here would notice. They are still the same scalar functions, called once per *distinct value*
through `_per_distinct_value` — which is both exactly equivalent and fast, because a production
frame holds far fewer distinct postcodes than rows and a batch of predictions shares one
`predicted_at` to the microsecond.

**What the fixture missed and the live table caught.** A 59-row fixture, one row per branch,
captured from the pre-refactor implementation, passed on the first run of the vectorised version.
It was still wrong. Replaying old against new over the real 11,268 rows showed the four component
columns matching exactly and the **composite differing on 703 rows by 0.1**.

The cause: `np.round(x, 1)` multiplies by ten, applies `rint`, and divides back; Python's
`round(x, 1)` converts the float exactly to decimal. The scaled value is not always exactly
representable, so the two disagree on values sitting on a boundary — and the composite lands on one
often, since a proximity score ending in `.5` against a weight of `0.30` is enough.

0.1 does not matter to a human reading the grid. It matters because the queue is **sorted** on that
column and a batch takes the top 25, so a tie broken the other way is a different restaurant
profiled at Gemini prices. `_round_like_python` restores the original semantics at all five
rounding sites.

**Verification.** After the fix, old and new agree on **all 901,440 values** — 16 anchor/preset
configurations × 5 columns × 11,268 rows — with identical ranked order in every configuration. The
fixture is kept as the offline guard and was mutation-checked (moving the 14-day staleness tier to
15 fails it, naming the case and the column).

The general lesson is the one this track keeps re-learning: a fixture proves the branches you
thought of. Floating-point equivalence is not a branch, and only real data at real scale surfaced
it.

**The cache.** `priority_for_current_frame` memoizes across reruns, keyed on a `data_version`
counter that only `set_enriched_frame` moves, with an `ast` test asserting no other site assigns
`df_enriched`. Deliberately *not* `@st.cache_data`: that hashes the frame's contents to build its
key, which for 11,268 rows costs about what the scoring saves. A cache over a frame the app mutates
is the same hazard `reset_selection_state` exists for, so the tests care more about every way it
must miss — new anchor, new preset, reloaded frame, same-sized frame with changed values, new day —
than about the hit.

| | before | after |
|---|---|---|
| Full scoring pass, 11,268 rows | 0.80s | **0.08s** |
| Rerun with nothing changed | 0.80s | **0.002s** |

**Related.** [D-18] is the other defect in this function, and the reason `first_present` exists.

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
| Model input features, before / after the Phase 6 retrain | **19 → 21** | 2026-09-23 |
| `pillar_value_rating` range in the trained model | **constant 0 → 0–8** | 2026-09-23 |
| `pillar_community_score` / `pillar_linguistic_score` / `pillar_culinary_score` | constant 0 → **0–6 / 0–7 / 0–6** | 2026-09-23 |
| `pillar_geo_specificity` in the trained model | constant 0 → **3 categories** | 2026-09-23 |
| `pillar_is_sit_down` / `pillar_establishment_type` | not read → **2 / 3 categories** | 2026-09-23 |
| `maps_rating` as the model sees it | min −1.0, 0 nulls → **min 2.1, 166 nulls** | 2026-09-23 |
| Retrain training population / unprofiled | 370 / **7** (all NULL postcode — D-16) | 2026-09-23 |
| Unprofiled rows with a NULL postcode | **126** of 8,494; 7 labelled | 2026-09-23 |
| Profiled rows carrying a `gemini_profiled_at` stamp | **2,767 of 2,767** | 2026-09-23 |
| Rows stale at the Phase 7 threshold of 180 days | **0**; first eligible 2027-03-22 | 2026-09-23 |
| Unprofiled rows, whole table / in-scope slice | **8,501** / 1,116; **0 labelled** | 2026-09-23 |
| Rows holding V1 `gemini_insights` text | **1,116** — all with no structured profile, no stamp, 1,090 in scope | 2026-09-23 |
| Rows with no coordinates / no coordinates and no placeable postcode | 8,977 / **106** | 2026-09-23 |
| Postcodes: NULL / empty string / junk | 126 / **0** / 2 (`WATERLOOVI`, `NE`) | 2026-09-23 |
| Unplaceable rows, proximity before → after | 14.7 (Trafalgar Square, 9.59 km) → **10.0, distance blank** | 2026-09-23 |
| Rows whose proximity score changes in Phase 8 | **106** — exactly the unplaceable ones | 2026-09-23 |
| Rows whose staleness score changes (D-18) | **1,051**: 1,020 from a wrong 100 down to their tier, 31 up to 100 | 2026-09-23 |
| Rows whose priority score changes in Phase 8 | **576** | 2026-09-23 |
| Unprofiled rows in the top 25 / 100 / 500 of the queue, before → after | 0 → **4** / 7 → **37** / 178 → **315** | 2026-09-23 |
| Top-25 / top-100 queue overlap, before vs. after | 4/25 / 43/100 | 2026-09-23 |
| `maps_found`: TRUE / FALSE / NULL | 2,263 / **243** / **8,762** | 2026-09-23 |
| Rows found on Maps but unrated | **0** — so the metric's number is unchanged, its meaning is not | 2026-09-23 |
| `-1` sentinels in `maps_rating` / `maps_reviews` / `price_level` / `match_score` | **0 / 0 / 0 / 0** | 2026-09-23 |
| Phase 9 split, re-run unchanged | 292 train / 77 holdout, means 2.613 / 2.299 — **identical to Phase 2** | 2026-09-23 |
| Production model features vs. `feature_select_list()` | **21 / 21 exact match**, label excluded | 2026-09-23 |
| Repaired tree MAE / RMSE / R² / Spearman | **0.737 / 1.320 / 0.518 / 0.689** | 2026-09-23 |
| Boosted tree, Phase 2 → Phase 9 | MAE 0.847 → **0.737** (−13%); ρ 0.545 → **0.689** | 2026-09-23 |
| `match_score`-only, Phase 2 → Phase 9 | MAE 0.697 → 0.679; ρ 0.603 → 0.645 (data moved, harness did not) | 2026-09-23 |
| Median absolute error, tree / `match_score` | **0.308** / 0.492 | 2026-09-23 |
| p90 absolute error, tree / `match_score` | 2.101 / **1.251** | 2026-09-23 |
| Holdout rows off by more than 2, tree / `match_score` | 8 / **5** of 77 | 2026-09-23 |
| Paired bootstrap, ΔMAE (tree − match), 20k resamples | +0.058, 95% CI [−0.073, +0.210], P(tree better) 0.21 | 2026-09-23 |
| Paired bootstrap, Δρ (tree − match) | +0.044, 95% CI [−0.055, +0.156], P(tree better) 0.79 | 2026-09-23 |
| Mean true rating of top-5 / top-10, tree | 5.60 / 5.20 | 2026-09-23 |
| Mean true rating of top-5 / top-10, `match_score` | **6.60 / 6.10** (oracle 7.00 / 6.70; overall 2.30) | 2026-09-23 |
| Predictions predating the Phase 6 retrain | **1,065 of 1,065**; 0 current | 2026-09-23 |
| Stale predictions profiled / unprofiled / labelled | 1,022 / **43** / 32 | 2026-09-23 |
| Stale predictions cleared | **1,065**; rows 11,268 and labels 411 unchanged | 2026-09-23 |
| Staleness tiers after the clear | **100.0 on all 11,268** — nothing is scored by the current model | 2026-09-23 |
| Unprofiled rows in the top 25 / 100 / 500, after the clear | **0** / 9 / 189 (was 4 / 37 / 315) | 2026-09-23 |
| V1 `gemini_insights` rows / in scope / labelled | 1,116 / 1,090 / **0** | 2026-09-23 |
| V1 rows that also hold a V2 structured profile | **0** — the two sets are exactly disjoint | 2026-09-23 |
| V1 text length, min / median / max | 126 / 1,379 / 3,421 characters; 1,116 distinct values | 2026-09-23 |
| V1 archive integrity | 1,116 = 1,116 rows, `BIT_XOR` hashes equal, 1,526,413 = 1,526,413 chars | 2026-09-23 |
| `fsa_master` columns after the drop | **43** (was 44); 11,268 rows, 411 labels unchanged | 2026-09-23 |
| `MASTER_BQ_SCHEMA` vs live table after the drop | 43 = 43, nothing named-but-absent | 2026-09-23 |
| `manual_review` distribution | `rejected` 10,869 / NULL 288 / `pending` 109 / `prending` 2 | 2026-09-23 |
| `manual_review = 'rejected'` that are `in_scope = TRUE` | **9,348**; 351 of them carry a `user_rating` | 2026-09-23 |
| Enrichment filter, 33-day window, old vs new | **153 → 146 rows**; the 7 dropped are all `in_scope = FALSE` | 2026-09-23 |
| `fsa_master` columns after both Phase 10 drops | **42** (was 44); 11,268 rows, 411 labels, 2,774 profiles unchanged | 2026-09-23 |
| Priority scoring, full pass over 11,268 rows | **0.80s → 0.08s**; warm rerun **0.002s** | 2026-09-24 |
| Old vs new scoring, live table | **0 mismatches / 901,440 values** (16 configs × 5 cols × 11,268 rows) | 2026-09-24 |
| `np.round` vs Python `round` on the composite | **703 of 11,268 rows** differed by 0.1 before `_round_like_python` | 2026-09-24 |
| Offline suite | **405 passed**, 300 subtests (was 345 at the Phase 10 checkpoint) | 2026-09-24 |

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
| Retrain, pulled forward into Phase 6 | 370 rows; JIT fires 7 Gemini calls, 0 Places, 0 postcodes | **< £0.05** — pending approval |
| **Legacy re-profile — withdrawn** | would have been 2,767 × `AI.GENERATE` | **avoided** |
| Orphaned scratch tables dropped | `recents`, `genairesults_temp`, 2 × `temp_update_reviews_*` | **£0** — done, 2026-09-23 |
| Phase 7 sweep, tokens | 1,116 × `gemini-3.8-flash` @ $0.75/$3.75 per 1M | **$7–$24** (£6–£19) |
| Phase 7 sweep, grounding | 1,116+ searches; 5,000/month free across Gemini 3.x, then $14/1,000 | **$0–$35** (£0–£28) |
| Phase 7 sweep, Places for the 975 without Maps | incurred anyway on first prediction, not by the sweep | **~$31** (£25), separate |
| **Phase 7 sweep, total** | | **$7–$59 / £6–£47, pending a measured pilot** |

| Phase 9 harness re-run | 2 throwaway models on 292 rows; 10.4 MiB dry-run, ~16 MiB executed | **< £0.01** — spent |
| Phase 9 stale-prediction `UPDATE` | 11.2 MiB over 11,268 rows | **< £0.01** |
| Re-scoring the 43 cleared rows with no profile | 43 × `AI.GENERATE` @ the Phase 6 measured rate | **~£0.30**, incurred only when the user asks |
| V1 archive table | 1,116 rows, 1.9 MiB scanned to build, ~1.5 MiB stored | **< £0.01/month** — spent |
| `DROP COLUMN gemini_insights` | metadata-only DDL | **£0** — spent |
| `manual_review` → `in_scope IS NOT FALSE` | 7 fewer `AI.GENERATE` calls per untargeted run | **small recurring saving** |
| `DROP COLUMN manual_review` | metadata-only DDL | **£0** — pending approval |

The only line item that needs its own approval is the Phase 7 sweep; everything up to Phase 9 is
pennies because the backfill turned out to be pure SQL.

The Phase 7 range is wide for two reasons that no amount of arithmetic will close: grounded calls
inject retrieved search results into the input, and `gemini-3.8-flash` bills thinking tokens as
output. Both are unobservable from the stored `.result`. `AI.GENERATE` returns `usageMetadata` in
its `full_response` field, which the current script discards -- a 20-row pilot costing roughly $0.15
would replace the range with a measured per-call figure before anything is committed.
