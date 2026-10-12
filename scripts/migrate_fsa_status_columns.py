"""Add and backfill FSA liveness tracking columns on `fsa_master`.

Columns added (all NULLABLE):
- `last_seen` (`DATE`): Last weekly fetch or API probe that observed this FHRSID live.
- `fsa_status` (`STRING`): `'active'` or `'delisted'`.
- `delisted_at` (`DATE`): Date when HTTP 404 was first observed on the FSA API.
- `superseded_by` (`STRING`): FHRSID of the live replacement establishment (re-listings only).

Backfills `fsa_status = 'active'` where NULL and `last_seen = first_seen` where NULL.
Strictly idempotent across repeated runs.

Usage:
    python -m scripts.migrate_fsa_status_columns             # dry-run validation
    python -m scripts.migrate_fsa_status_columns --execute   # execute DDL + backfill in BigQuery
"""
import argparse
import logging
import subprocess
from typing import Optional

from google.api_core import exceptions as gcp_exceptions
from google.cloud import bigquery
from google.oauth2 import credentials as oauth2_credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

FSA_STATUS_COLUMNS = (
    ("last_seen", "DATE"),
    ("fsa_status", "STRING"),
    ("delisted_at", "DATE"),
    ("superseded_by", "STRING"),
)


def build_add_columns_sql(bq_path: str) -> str:
    """DDL statement adding the 4 FSA liveness columns if not already present."""
    clauses = ",\n".join(
        f"ADD COLUMN IF NOT EXISTS {col} {bq_type}"
        for col, bq_type in FSA_STATUS_COLUMNS
    )
    return f"ALTER TABLE `{bq_path}`\n{clauses}"


def build_backfill_sql(bq_path: str) -> str:
    """Idempotent DML backfilling `fsa_status='active'` and `last_seen=first_seen` where NULL."""
    return f"""UPDATE `{bq_path}`
SET
  fsa_status = COALESCE(fsa_status, 'active'),
  last_seen = COALESCE(last_seen, first_seen)
WHERE fsa_status IS NULL
   OR (last_seen IS NULL AND first_seen IS NOT NULL)"""


def build_summary_sql(bq_path: str) -> str:
    """Summary query reporting liveness column population counts."""
    return f"""SELECT
  COUNT(*) AS total_rows,
  COUNTIF(fsa_status = 'active') AS active_rows,
  COUNTIF(fsa_status = 'delisted') AS delisted_rows,
  COUNTIF(fsa_status IS NULL) AS null_status_rows,
  COUNTIF(last_seen IS NOT NULL) AS last_seen_populated,
  COUNTIF(delisted_at IS NOT NULL) AS delisted_at_populated,
  COUNTIF(superseded_by IS NOT NULL) AS superseded_rows
FROM `{bq_path}`"""


def _gcloud_token_client(project_id: str) -> bigquery.Client:
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token", "filipegracio@google.com"],
        text=True,
    ).strip()
    creds = oauth2_credentials.Credentials(token=token)
    return bigquery.Client(project=project_id, credentials=creds)


def run_migration(
    bq_path: str = DEFAULT_BQ_PATH,
    execute: bool = False,
    client: Optional[bigquery.Client] = None,
) -> dict:
    """Validate or execute the FSA liveness schema migration and initial backfill."""
    project_id = bq_path.split(".")[0]
    bq_client = client or bigquery.Client(project=project_id)

    add_sql = build_add_columns_sql(bq_path)
    backfill_sql = build_backfill_sql(bq_path)
    summary_sql = build_summary_sql(bq_path)

    if not execute:
        try:
            job = bq_client.query(
                add_sql,
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
            )
        except gcp_exceptions.Forbidden:
            if client is not None:
                raise
            bq_client = _gcloud_token_client(project_id)
            job = bq_client.query(
                add_sql,
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
            )
        logger.info(
            "[DRY RUN] Validated ALTER TABLE DDL (%s bytes).\n%s\n%s",
            job.total_bytes_processed,
            add_sql,
            backfill_sql,
        )
        return {"dry_run": True, "bytes_processed": job.total_bytes_processed}

    try:
        bq_client.query(add_sql).result()
    except gcp_exceptions.Forbidden:
        if client is not None:
            raise
        bq_client = _gcloud_token_client(project_id)
        bq_client.query(add_sql).result()

    logger.info("Ensured columns `last_seen`, `fsa_status`, `delisted_at`, `superseded_by` exist.")

    before = dict(list(bq_client.query(summary_sql).result())[0])
    logger.info("Before backfill: %s", before)

    backfill_job = bq_client.query(backfill_sql)
    backfill_job.result()
    backfilled_rows = backfill_job.num_dml_affected_rows or 0
    logger.info("Backfilled %d row(s) with fsa_status='active' and last_seen=first_seen.", backfilled_rows)

    after = dict(list(bq_client.query(summary_sql).result())[0])
    logger.info("After backfill: %s", after)

    if after["null_status_rows"] != 0:
        raise RuntimeError(f"Expected 0 null_status_rows after backfill, got: {after}")

    return {
        "dry_run": False,
        "backfilled_rows": backfilled_rows,
        "before": before,
        "after": after,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH, help="Full BigQuery table path")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the DDL/DML migration in BigQuery. Without this, only dry-run validation runs.",
    )
    args = parser.parse_args()
    run_migration(bq_path=args.bq_path, execute=args.execute)
