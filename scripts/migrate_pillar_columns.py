"""Add the typed pillar columns to `fsa_master`. Additive and idempotent.

This is the expand half of an expand/contract migration. Every column is
nullable and nothing reads them yet, so applying this changes no behaviour --
Phase 5 fills them, Phase 6 writes them, Phase 7 switches readers. Nothing is
dropped from the live table before Phase 10.

    python -m scripts.migrate_pillar_columns                     # validate, run nothing
    python -m scripts.migrate_pillar_columns --bq_path ...backup_20260923 --execute
    python -m scripts.migrate_pillar_columns --execute           # production

Dry run is the default, matching `scripts/recon_pipeline_state.py` rather than
the older `migrate_*` scripts, which execute unless told otherwise. This one
writes to the table holding the only copy of 411 hand-entered labels.

**Ordering.** This must run before `MASTER_BQ_SCHEMA` (`app/services/bq_utils.py`)
is updated to list these columns. That constant is the load schema for the
weekly cron's `append_to_bigquery`; if it names a column the live table lacks,
the scheduled ingest fails. The reverse order is safe -- a table with columns
the load schema omits simply gets NULLs -- and nothing in the repo reconciles
the two automatically.
"""
import argparse
import logging

from google.cloud import bigquery

from app.core.pillar_schema import NON_JSON_COLUMNS, PILLAR_FIELDS
from app.services.bq_utils import MASTER_BQ_SCHEMA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

# Generated, never retyped. A second hand-written copy of the pillar schema
# here would be the fifth, and re-create the class of bug Phase 3 closed.
NEW_COLUMNS = tuple(
    [(field.column, field.bq_type) for field in PILLAR_FIELDS] + list(NON_JSON_COLUMNS)
)

# The columns that exist before this migration. Held invariant across it, and
# named explicitly so the fingerprint cannot drift as columns are added.
PRE_EXISTING_COLUMNS = tuple(field.name for field in MASTER_BQ_SCHEMA)


def build_ddl_statements(bq_path: str) -> list:
    """One `ADD COLUMN IF NOT EXISTS` per new column.

    Separate statements rather than one multi-column `ALTER`: if the run is
    interrupted, re-running it skips what already landed instead of failing on
    the first duplicate.
    """
    return [
        f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS {name} {bq_type}"
        for name, bq_type in NEW_COLUMNS
    ]


def build_fingerprint_query(bq_path: str) -> str:
    """Row count, label count, and a content hash over the pre-existing columns.

    `ADD COLUMN` is metadata-only in BigQuery, so this should be identical
    before and after. It is here because "should be" is not evidence, and
    because `user_rating` cannot be regenerated if it is wrong.

    The hash covers `PRE_EXISTING_COLUMNS` explicitly rather than the whole row:
    `TO_JSON_STRING(t)` would pick up the newly added NULL columns and differ
    between the two readings for a reason that has nothing to do with the data.
    """
    struct_fields = ", ".join(PRE_EXISTING_COLUMNS)
    # `row_count`, not `rows`: ROWS is a reserved keyword in BigQuery (window
    # frames), and the alias is a syntax error. Caught by the dry run.
    return f"""SELECT
  COUNT(*) AS row_count,
  COUNT(user_rating) AS labels,
  BIT_XOR(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT({struct_fields})))) AS fingerprint
FROM `{bq_path}`"""


def _fingerprint(client: bigquery.Client, bq_path: str, when: str) -> tuple:
    row = list(client.query(build_fingerprint_query(bq_path)).result())[0]
    logger.info(
        f"Fingerprint {when}: rows={row.row_count} labels={row.labels} hash={row.fingerprint}"
    )
    return (row.row_count, row.labels, row.fingerprint)


def run_migration(bq_path: str = DEFAULT_BQ_PATH, execute: bool = False) -> None:
    """Add the columns, verifying the existing data is untouched either side."""
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    statements = build_ddl_statements(bq_path)

    logger.info(f"Target: {bq_path}")
    logger.info(f"Adding {len(statements)} nullable columns.")

    if not execute:
        # Validating is the point. Printing SQL that nobody parsed would let a
        # typo in a type name survive the approval it is meant to inform.
        for stmt in statements:
            client.query(stmt, job_config=bigquery.QueryJobConfig(dry_run=True))
            logger.info(f"[DRY RUN] valid: {stmt}")
        client.query(
            build_fingerprint_query(bq_path),
            job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
        )
        logger.info("[DRY RUN] valid: fingerprint query")
        logger.info("Dry run complete. Nothing was written. Re-run with --execute.")
        return

    before = _fingerprint(client, bq_path, "before")

    for stmt in statements:
        logger.info(f"Executing: {stmt}")
        # Deliberately unguarded. `migrate_to_in_scope_workflow.py` downgrades a
        # DDL failure to a warning and carries on; a half-applied schema here
        # would let Phase 5's backfill target columns that do not exist.
        client.query(stmt).result()

    after = _fingerprint(client, bq_path, "after")

    if before != after:
        raise RuntimeError(
            f"Existing data changed across an additive migration: {before} -> {after}. "
            f"Restore from the snapshot before going further."
        )

    logger.info(f"Added {len(statements)} columns. Existing data verified unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add the typed pillar columns to the FSA master table (additive, idempotent)"
    )
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH,
                        help="Full BigQuery table path (project.dataset.table)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run the DDL. Without this, statements are only validated.")
    args = parser.parse_args()

    run_migration(bq_path=args.bq_path, execute=args.execute)
