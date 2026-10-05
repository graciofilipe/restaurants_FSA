"""Backfill `rating_source = 'desk'` on legacy obvious-negative labels in `fsa_master`.

Before `rating_source` was introduced (`'visited'` vs. `'desk'`), 377 historical
`user_rating` labels were entered with `rating_source IS NULL`. Most of those
are fast-food chains, takeaways, non-restaurant businesses, or explicit `1-3`
desk rejections, while a small subset (`~20` sit-down restaurants rated `>= 4`)
are ambiguous and must remain `NULL` for manual UI triage.

    python -m scripts.backfill_rating_source             # dry-run validation
    python -m scripts.backfill_rating_source --execute   # execute UPDATE in BigQuery
"""
import argparse
import logging

from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"


def build_backfill_where_clause() -> str:
    """Predicate matching legacy obvious-negative labels that are unambiguously
    desk triage rejections (`rating_source IS NULL`)."""
    return (
        "user_rating IS NOT NULL "
        "AND rating_source IS NULL "
        "AND ("
        "COALESCE(in_scope, TRUE) IS FALSE "
        "OR COALESCE(pillar_is_sit_down, TRUE) IS FALSE "
        "OR COALESCE(pillar_establishment_type, 'RESTAURANT_DINING') != 'RESTAURANT_DINING' "
        "OR user_rating <= 3"
        ")"
    )


def build_update_sql(bq_path: str) -> str:
    """DML statement tagging legacy obvious negatives as `rating_source = 'desk'`."""
    return f"""UPDATE `{bq_path}`
SET rating_source = 'desk'
WHERE {build_backfill_where_clause()}"""


def build_summary_sql(bq_path: str) -> str:
    """Breakdown of `rating_source` and invariant checks over `user_rating`."""
    return f"""SELECT
  COUNT(*) AS total_rows,
  COUNT(user_rating) AS labeled_rows,
  COALESCE(SUM(user_rating), 0) AS sum_user_rating,
  COUNTIF(user_rating IS NOT NULL AND rating_source = 'visited') AS visited_count,
  COUNTIF(user_rating IS NOT NULL AND rating_source = 'desk') AS desk_count,
  COUNTIF(user_rating IS NOT NULL AND rating_source IS NULL) AS null_source_count,
  COUNTIF({build_backfill_where_clause()}) AS eligible_for_desk_backfill
FROM `{bq_path}`"""


def run_backfill(bq_path: str = DEFAULT_BQ_PATH, execute: bool = False) -> dict:
    """Validate or execute the `rating_source = 'desk'` backfill."""
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    summary_sql = build_summary_sql(bq_path)
    update_sql = build_update_sql(bq_path)

    if not execute:
        client.query(
            summary_sql,
            job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
        )
        job = client.query(
            update_sql,
            job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
        )
        logger.info(
            f"[DRY RUN] Validated summary & UPDATE queries ({job.total_bytes_processed} bytes)."
        )
        return {"dry_run": True, "bytes_processed": job.total_bytes_processed}

    before = dict(list(client.query(summary_sql).result())[0])
    logger.info(f"Before backfill: {before}")

    update_job = client.query(update_sql)
    update_job.result()
    affected = update_job.num_dml_affected_rows or 0

    after = dict(list(client.query(summary_sql).result())[0])
    logger.info(f"After backfill (updated {affected} rows): {after}")

    if (
        before["labeled_rows"] != after["labeled_rows"]
        or before["sum_user_rating"] != after["sum_user_rating"]
        or before["visited_count"] != after["visited_count"]
    ):
        raise RuntimeError(
            f"Label invariant violated during rating_source backfill: {before} -> {after}"
        )

    return {"dry_run": False, "updated_rows": affected, "before": before, "after": after}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH, help="Full BigQuery table path")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the UPDATE in BigQuery. Without this, only dry-run validation runs.",
    )
    args = parser.parse_args()
    run_backfill(bq_path=args.bq_path, execute=args.execute)
