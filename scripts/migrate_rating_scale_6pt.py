"""Migrate `user_rating` in `fsa_master` to the canonical 1-6 scale.

Re-derives `user_rating` losslessly from `user_rating_10pt_backup` (`INT64`) where
present:
    new_rating = CAST(ROUND(1 + (user_rating_10pt_backup - 1) * 5.0 / 9.0) AS INT64)

Mapping across 1..10 -> 1..6:
    1 -> 1 | 2, 3 -> 2 | 4, 5 -> 3 | 6, 7 -> 4 | 8, 9 -> 5 | 10 -> 6

For rows rated on the interim 1-7 scale (where `user_rating_10pt_backup IS NULL`),
preserves the 7-point value in `user_rating_7pt_backup` (`INT64`) before mapping:
    new_rating = CAST(ROUND(1 + (user_rating_7pt_backup - 1) * 5.0 / 6.0) AS INT64)

Because both branches read from immutable backup columns (`user_rating_10pt_backup`
and `user_rating_7pt_backup`), the migration is strictly idempotent and safe against
repeated execution.

Usage:
    python -m scripts.migrate_rating_scale_6pt             # dry-run validation
    python -m scripts.migrate_rating_scale_6pt --execute   # execute migration in BigQuery
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


def convert_10pt_to_6pt(score_10pt: int) -> int:
    """Convert an integer rating on [1, 10] to [1, 6] via affine rescaling."""
    if score_10pt < 1 or score_10pt > 10:
        raise ValueError(f"Expected 1-10 rating, got {score_10pt}")
    return int(round(1 + (score_10pt - 1) * 5.0 / 9.0))


def convert_7pt_to_6pt(score_7pt: int) -> int:
    """Convert an integer rating on [1, 7] to [1, 6] via affine rescaling."""
    if score_7pt < 1 or score_7pt > 7:
        raise ValueError(f"Expected 1-7 rating, got {score_7pt}")
    return int(round(1 + (score_7pt - 1) * 5.0 / 6.0))


def build_add_backup_column_sql(bq_path: str) -> str:
    """DDL statement ensuring `user_rating_10pt_backup` exists on `fsa_master`."""
    return f"""ALTER TABLE `{bq_path}`
ADD COLUMN IF NOT EXISTS user_rating_10pt_backup INT64"""


def build_add_7pt_backup_column_sql(bq_path: str) -> str:
    """DDL statement ensuring `user_rating_7pt_backup` exists on `fsa_master`."""
    return f"""ALTER TABLE `{bq_path}`
ADD COLUMN IF NOT EXISTS user_rating_7pt_backup INT64"""


def build_backfill_7pt_backup_sql(bq_path: str) -> str:
    """DML statement copying interim 1-7 `user_rating` into `user_rating_7pt_backup`
    only for rows where `user_rating_10pt_backup IS NULL` and `user_rating_7pt_backup IS NULL`."""
    return f"""UPDATE `{bq_path}`
SET user_rating_7pt_backup = user_rating
WHERE user_rating IS NOT NULL
  AND user_rating_10pt_backup IS NULL
  AND user_rating_7pt_backup IS NULL"""


def build_convert_rating_sql(bq_path: str) -> str:
    """Idempotent DML statement computing 1-6 `user_rating` from `user_rating_10pt_backup`
    (when present) or `user_rating_7pt_backup` (for rows rated after the 10pt backup)."""
    return f"""UPDATE `{bq_path}`
SET user_rating = CASE
  WHEN user_rating_10pt_backup IS NOT NULL THEN
    CAST(ROUND(1 + (user_rating_10pt_backup - 1) * 5.0 / 9.0) AS INT64)
  WHEN user_rating_7pt_backup IS NOT NULL THEN
    CAST(ROUND(1 + (user_rating_7pt_backup - 1) * 5.0 / 6.0) AS INT64)
  ELSE user_rating
END
WHERE user_rating_10pt_backup IS NOT NULL
   OR user_rating_7pt_backup IS NOT NULL"""


def build_summary_sql(bq_path: str) -> str:
    """Summary query verifying row counts and 1-6 rating bounds."""
    return f"""SELECT
  COUNT(*) AS total_rows,
  COUNT(user_rating) AS labeled_rows,
  MIN(user_rating) AS min_user_rating,
  MAX(user_rating) AS max_user_rating,
  COUNTIF(user_rating > 6) AS rows_above_6,
  COUNTIF(user_rating IS NOT NULL AND rating_source = 'visited') AS visited_count,
  COUNTIF(user_rating IS NOT NULL AND rating_source = 'desk') AS desk_count,
  COUNTIF(user_rating IS NOT NULL AND rating_source IS NULL) AS null_source_count
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
    """Validate or execute the 1-6 rating scale migration."""
    project_id = bq_path.split(".")[0]
    bq_client = client or bigquery.Client(project=project_id)

    summary_sql = build_summary_sql(bq_path)
    add_10pt_col_sql = build_add_backup_column_sql(bq_path)
    add_7pt_col_sql = build_add_7pt_backup_column_sql(bq_path)
    backfill_7pt_sql = build_backfill_7pt_backup_sql(bq_path)
    convert_sql = build_convert_rating_sql(bq_path)

    if not execute:
        try:
            job = bq_client.query(
                summary_sql,
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
            )
        except gcp_exceptions.Forbidden:
            if client is not None:
                raise
            bq_client = _gcloud_token_client(project_id)
            job = bq_client.query(
                summary_sql,
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
            )
        logger.info(
            f"[DRY RUN] Validated summary query ({job.total_bytes_processed} bytes)."
        )
        return {"dry_run": True, "bytes_processed": job.total_bytes_processed}

    try:
        before = dict(list(bq_client.query(summary_sql).result())[0])
    except gcp_exceptions.Forbidden:
        if client is not None:
            raise
        bq_client = _gcloud_token_client(project_id)
        before = dict(list(bq_client.query(summary_sql).result())[0])

    logger.info(f"Before 1-6 migration: {before}")

    bq_client.query(add_10pt_col_sql).result()
    bq_client.query(add_7pt_col_sql).result()
    logger.info("Ensured columns `user_rating_10pt_backup` and `user_rating_7pt_backup` exist.")

    backfill_job = bq_client.query(backfill_7pt_sql)
    backfill_job.result()
    backed_up_7pt_rows = backfill_job.num_dml_affected_rows or 0
    logger.info(f"Backed up {backed_up_7pt_rows} post-10pt rating row(s) to `user_rating_7pt_backup`.")

    convert_job = bq_client.query(convert_sql)
    convert_job.result()
    converted_rows = convert_job.num_dml_affected_rows or 0
    logger.info(f"Converted {converted_rows} rating row(s) to 1-6 scale.")

    after = dict(list(bq_client.query(summary_sql).result())[0])
    logger.info(f"After 1-6 migration: {after}")

    if (
        before["labeled_rows"] != after["labeled_rows"]
        or before["visited_count"] != after["visited_count"]
        or before["desk_count"] != after["desk_count"]
        or after.get("rows_above_6", 0) != 0
        or (after["labeled_rows"] > 0 and (after["min_user_rating"] < 1 or after["max_user_rating"] > 6))
    ):
        raise RuntimeError(
            f"Invariant violated during 1-6 rating scale migration: {before} -> {after}"
        )

    return {
        "dry_run": False,
        "backed_up_7pt_rows": backed_up_7pt_rows,
        "converted_rows": converted_rows,
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
