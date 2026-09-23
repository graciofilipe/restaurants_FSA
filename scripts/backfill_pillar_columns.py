"""Fill the Phase 4 columns from data the table already holds. Pure SQL, no model calls.

Phase 4 added 17 empty columns. This fills them from `gemini_insights_structured`,
converts the Maps `-1` sentinel into an honest found/not-found pair, and re-derives
`in_scope` from the pillar the original triage migration failed to read (D13).

    python -m scripts.backfill_pillar_columns                      # preview, write nothing
    python -m scripts.backfill_pillar_columns --bq_path ...backup_20260923 --execute
    python -m scripts.backfill_pillar_columns --execute            # production

Five UPDATEs, in a fixed order, all read-modify on one table. No Gemini, no Places,
no new rows: everything written here is already in the table in a less usable form.

**Two ordering rules, both load-bearing.**

Within the run, `in_scope` is derived from `pillar_is_sit_down`, so it must come
after the statement that fills it -- against an empty column it would set every
profiled row's `in_scope` to NULL. And `maps_hit` must precede `maps_miss`,
because the miss statement erases the `-1` the hit statement selects against.

Across the track, the *code* change that retires the sentinel (Phase 6, moving
three guards from `maps_rating IS NULL` to `maps_lookup_at IS NULL`) must deploy
strictly after this runs. Ship it first and `maps_lookup_at IS NULL` is true for
all 11,268 rows, which makes every row eligible for a paid Places lookup.

**On `in_scope` and human decisions.** The re-derivation deliberately skips any
row carrying a `user_rating` or `rating_source`. 214 of the 369 trainable labels
sit on rows the profiler calls not-sit-down; re-deriving those would overwrite a
human triage decision with a model's opinion and drop the training set to 157.
The guard costs little -- 1,263 of the 1,479 disagreements are still corrected.
"""
import argparse
import logging

from google.cloud import bigquery

from app.core.pillar_schema import NON_JSON_COLUMNS, PILLAR_FIELDS, sql_extract

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

# Dated in the past rather than CURRENT_TIMESTAMP(): this timestamp takes over
# the do-not-retry role the `-1` sentinel plays today, and we genuinely do not
# know when these lookups happened. A backdated value means a future staleness
# sweep re-checks them rather than trusting a lookup it never saw.
MAPS_BACKFILL_LOOKUP_AT = "TIMESTAMP('2026-01-01 00:00:00 UTC')"

# Rows with a profile to read. `gemini_insights_structured` is the raw JSON the
# merge writes; 2,767 of 11,268 rows have one.
_HAS_PROFILE = "gemini_insights_structured IS NOT NULL"

# A row the user has touched. Their judgement outranks the profiler's.
_UNTOUCHED_BY_HUMAN = "user_rating IS NULL AND rating_source IS NULL"


def _update(bq_path: str, set_clause: str, where: str) -> str:
    return f"UPDATE `{bq_path}`\nSET {set_clause}\nWHERE {where}"


def _count(bq_path: str, where: str) -> str:
    return f"SELECT COUNT(*) AS affected FROM `{bq_path}` WHERE {where}"


def build_pillar_backfill(bq_path: str) -> str:
    """Fill the 14 typed columns from the stored JSON.

    Generated from `PILLAR_FIELDS` via `sql_extract`, so the paths here are the
    same ones the parser and the conformance check use. A hand-written path list
    would be the fifth copy, and the discrepancy between the previous four is D2.

    No `IFNULL`. A profile that omits a score leaves NULL, because a column that
    cannot distinguish "the model said zero" from "the model said nothing" is
    what made D2 invisible for the life of the model.
    """
    assignments = ",\n    ".join(
        f"{field.column} = {sql_extract(field, 'gemini_insights_structured')}"
        for field in PILLAR_FIELDS
    )
    return _update(bq_path, assignments, _HAS_PROFILE)


def build_profiled_at_backfill(bq_path: str) -> str:
    """Treat every existing profile as fresh, following `migrate_predicted_at.py`.

    The alternative -- leaving these NULL so they look never-profiled -- would
    queue all 2,767 for a re-profile the moment Phase 7's staleness check ships.
    In a track whose headline defect is paying twice for Gemini, that is a
    self-inflicted bill.

    `gemini_profiled_at IS NULL` keeps it idempotent: a second run must not slide
    the staleness clock forward and postpone every future refresh.
    """
    return _update(
        bq_path,
        "gemini_profiled_at = CURRENT_TIMESTAMP()",
        f"{_HAS_PROFILE} AND gemini_profiled_at IS NULL",
    )


def build_maps_hit_backfill(bq_path: str) -> str:
    """Record a successful Places lookup, leaving the rating itself alone.

    `maps_rating IS NOT NULL AND maps_rating > 0` rather than just `> 0`: the
    8,762 rows never looked up must keep a NULL `maps_lookup_at`, or Phase 6's
    guard reads them as already tried.
    """
    return _update(
        bq_path,
        f"maps_found = TRUE,\n    maps_lookup_at = {MAPS_BACKFILL_LOOKUP_AT}",
        "maps_rating IS NOT NULL AND maps_rating > 0 AND maps_lookup_at IS NULL",
    )


def build_maps_miss_backfill(bq_path: str) -> str:
    """Retire the `-1` sentinel (D4/R3) without losing what it encoded.

    `-1` means two things at once today: "do not retry this" and, to
    `calculate_restaurant_priority`, a quality prior of zero -- worse than a
    genuinely unknown restaurant scores. Splitting it into `maps_found = FALSE`
    plus a non-NULL `maps_lookup_at` keeps the first meaning and drops the
    second: these 243 rows move from a 0 prior to the neutral 50.
    """
    return _update(
        bq_path,
        "maps_found = FALSE,\n    maps_lookup_at = " + MAPS_BACKFILL_LOOKUP_AT
        + ",\n    maps_rating = NULL,\n    maps_reviews = NULL",
        "maps_rating = -1",
    )


def build_in_scope_rederivation(bq_path: str) -> str:
    """D13. Re-derive scope from the pillar the original migration never read.

    `migrate_to_in_scope_workflow.py` gated on
    `'$.6_establishment_integrity_is_sit_down_restaurant'` -- convention B, a
    path that has never resolved -- so that branch never fired and `in_scope`
    was assigned from `maps_types` alone. 1,479 profiled rows disagree with
    their own profile.

    Three guards on what this may touch:
      * `pillar_is_sit_down IS NOT NULL` -- 8,501 rows were never profiled and
        one profile is unparseable; there is nothing to derive from.
      * `IS DISTINCT FROM` -- only rows whose value would actually change.
      * no `user_rating`, no `rating_source` -- see the module docstring.
    """
    return _update(
        bq_path,
        "in_scope = pillar_is_sit_down",
        f"pillar_is_sit_down IS NOT NULL\n  AND in_scope IS DISTINCT FROM pillar_is_sit_down"
        f"\n  AND {_UNTOUCHED_BY_HUMAN}",
    )


def build_all_statements(bq_path: str) -> list:
    """The five statements as (label, update_sql, count_sql), in execution order.

    The count query carries the same predicate as its UPDATE, so the preview
    cannot drift from what runs -- estimate-and-behaviour divergence is the
    exact shape of D1.
    """
    specs = [
        ('pillars', build_pillar_backfill, _HAS_PROFILE),
        ('profiled_at', build_profiled_at_backfill,
         f"{_HAS_PROFILE} AND gemini_profiled_at IS NULL"),
        ('maps_hit', build_maps_hit_backfill,
         "maps_rating IS NOT NULL AND maps_rating > 0 AND maps_lookup_at IS NULL"),
        ('maps_miss', build_maps_miss_backfill, "maps_rating = -1"),
        ('in_scope', build_in_scope_rederivation,
         f"pillar_is_sit_down IS NOT NULL"
         f"\n  AND in_scope IS DISTINCT FROM pillar_is_sit_down"
         f"\n  AND {_UNTOUCHED_BY_HUMAN}"),
    ]
    return [(label, builder(bq_path), _count(bq_path, where)) for label, builder, where in specs]


def build_validation_query(bq_path: str) -> str:
    """Per-column coverage and distinct-value counts. Read-only.

    Distinct counts, not just coverage: a feature column filled entirely with
    the same value is exactly as dead as an empty one, and that is the condition
    this track exists to end. Anything reading 1 here is a failed backfill.
    """
    lines = [f"  COUNT(*) AS total_rows"]
    for field in PILLAR_FIELDS:
        lines.append(f"  COUNT({field.column}) AS filled_{field.column}")
        lines.append(f"  COUNT(DISTINCT {field.column}) AS distinct_{field.column}")
    for name, _ in NON_JSON_COLUMNS:
        lines.append(f"  COUNT({name}) AS filled_{name}")
    lines.append("  COUNTIF(maps_rating = -1) AS surviving_sentinels")
    lines.append("  COUNTIF(in_scope) AS in_scope_rows")
    lines.append("  COUNT(user_rating) AS labels")
    return "SELECT\n" + ",\n".join(lines) + f"\nFROM `{bq_path}`"


def _preview(client: bigquery.Client, label: str, count_sql: str) -> int:
    row = list(client.query(count_sql).result())[0]
    logger.info(f"  {label}: {row.affected} rows would be updated")
    return row.affected


def run_backfill(bq_path: str = DEFAULT_BQ_PATH, execute: bool = False) -> None:
    """Preview, then optionally apply, the five statements in order."""
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    statements = build_all_statements(bq_path)

    logger.info(f"Target: {bq_path}")
    logger.info(f"{len(statements)} statements, in dependency order.")

    if not execute:
        # Counts are read against the table as it stands now. `in_scope` reads
        # a column an earlier statement in this same run fills, so its preview
        # is 0 until the pillar backfill has actually landed -- run this against
        # the snapshot copy after backfilling it to see the real figure.
        for label, update_sql, count_sql in statements:
            client.query(update_sql, job_config=bigquery.QueryJobConfig(dry_run=True))
            logger.info(f"[DRY RUN] valid: {label}")
            _preview(client, label, count_sql)
        logger.info("Dry run complete. Nothing was written. Re-run with --execute.")
        return

    for label, update_sql, count_sql in statements:
        _preview(client, label, count_sql)
        logger.info(f"Executing: {label}")
        # Unguarded, like the Phase 4 migration. The statements are dependent --
        # `in_scope` derives from a column `pillars` fills -- so carrying on past
        # a failure would take the whole profiled set out of scope.
        job = client.query(update_sql)
        job.result()
        logger.info(f"  {label}: {job.num_dml_affected_rows} rows updated")

    logger.info("Backfill complete. Run the validation query to confirm coverage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill the typed pillar columns, Maps found/lookup flags, and in_scope"
    )
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH,
                        help="Full BigQuery table path (project.dataset.table)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run the UPDATEs. Without this, they are only validated.")
    parser.add_argument("--validate", action="store_true",
                        help="Print per-column coverage instead of running the backfill.")
    args = parser.parse_args()

    if args.validate:
        bq_client = bigquery.Client(project=args.bq_path.split(".")[0])
        result = list(bq_client.query(build_validation_query(args.bq_path)).result())[0]
        for key, value in result.items():
            logger.info(f"{key}: {value}")
    else:
        run_backfill(bq_path=args.bq_path, execute=args.execute)
