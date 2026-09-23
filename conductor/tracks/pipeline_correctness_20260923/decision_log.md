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

## Measurements

*Populated by Phase 0. Empty until recon runs.*

| Measurement | Value | Date |
|---|---|---|
| Snapshot row count | _pending_ | |
| Snapshot `COUNT(user_rating)` | _pending_ | |
| Observed top-level key census | _pending_ | |
| Shape stable across rows? | _pending_ | |
| `COUNT(*) WHERE gemini_insights IS NOT NULL` (expect 0) | _pending_ | |
| Rows with `maps_rating = -1` | _pending_ | |
| Rows with lat/lon | _pending_ | |
| `in_scope` rows assigned by `maps_types` alone | _pending_ | |
| Baseline model MAE/RMSE | _pending_ | |
| `match_score`-only baseline | _pending_ | |
