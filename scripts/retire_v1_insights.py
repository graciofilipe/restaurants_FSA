"""Archive the V1 `gemini_insights` text, then drop the column.

`gemini_insights` holds free-text evaluations written by the profiler that
preceded the structured V2 JSON -- 1,116 rows, median 1,379 characters, all
distinct. They are not a duplicate of anything: the rows carrying them have no
`gemini_insights_structured` at all, the two sets are exactly disjoint, and
none of the 1,116 has ever been given a `user_rating`. Nothing in the app reads
the column, nothing scores it, and the V2 merge nulls it on write.

So it is dead weight with one irreplaceable property, and the two halves of
this script reflect that: keep the text somewhere permanent, then take the
column out of the live table.

    python -m scripts.retire_v1_insights --archive               # validate, run nothing
    python -m scripts.retire_v1_insights --archive --execute     # write the archive table
    python -m scripts.retire_v1_insights --drop                  # validate, run nothing
    python -m scripts.retire_v1_insights --drop --execute        # ALTER TABLE DROP COLUMN

**The drop is gated on the archive.** It refuses to run unless the archive
table exists and holds exactly as many rows as the source still has text for.
A drop is not reversible from anything but the Phase 0 snapshot, and the
snapshot is scheduled to be deleted at the end of the track.

**Ordering.** The code must stop naming the column before it disappears.
`MASTER_BQ_SCHEMA` is the load schema for the weekly cron's
`append_to_bigquery`; if it lists a column the live table lacks, the scheduled
ingest fails. So: land and deploy the code change, then drop.
"""
import argparse
import logging

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
DEFAULT_ARCHIVE_PATH = (
    "filipegracio-ai-learning.filipegracio_fsa_restaurants.gemini_insights_v1_archive_20260923")


def build_source_count_sql(bq_path: str) -> str:
    """How many rows still carry V1 text. Zero means the drop already happened."""
    return f"SELECT COUNTIF(gemini_insights IS NOT NULL) AS rows_with_text FROM `{bq_path}`"


def build_archive_sql(bq_path: str, archive_path: str) -> str:
    """The text, plus enough columns to know whose it is.

    Plain `CREATE TABLE`: `OR REPLACE` would let a re-run after the drop
    overwrite a good archive with an empty one, and `IF NOT EXISTS` would make
    that same re-run look like a success.
    """
    return f"""CREATE TABLE `{archive_path}` AS
SELECT fhrsid, businessname, postcode, first_seen, gemini_insights
FROM `{bq_path}`
WHERE gemini_insights IS NOT NULL"""


def build_drop_sql(bq_path: str) -> str:
    """One column, named in full. `gemini_insights_structured` stays."""
    return f"ALTER TABLE `{bq_path}` DROP COLUMN gemini_insights"


def _source_rows_with_text(client, bq_path: str) -> int:
    rows = list(client.query(build_source_count_sql(bq_path)).result())
    return rows[0].rows_with_text


def _archive_rows(client, archive_path: str):
    try:
        return client.get_table(archive_path).num_rows
    except NotFound:
        return None


def run_retirement(bq_path: str = DEFAULT_BQ_PATH,
                   archive_path: str = DEFAULT_ARCHIVE_PATH,
                   archive: bool = False,
                   drop: bool = False,
                   execute: bool = False) -> dict:
    """Archive, drop, or validate either. Dry run unless `execute` is set."""
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    state = {}

    source_rows = _source_rows_with_text(client, bq_path)
    state['source_rows_with_text'] = source_rows
    logger.info(f"{source_rows} rows still carry V1 `gemini_insights` text.")

    if archive:
        sql = build_archive_sql(bq_path, archive_path)
        logger.info(f"--- archive ---\n{sql}\n")
        if execute:
            client.query(sql).result()
            archived = _archive_rows(client, archive_path)
            state['archived_rows'] = archived
            logger.info(f"Archive `{archive_path}` created with {archived} rows.")
        else:
            job = client.query(sql, job_config=bigquery.QueryJobConfig(
                dry_run=True, use_query_cache=False))
            logger.info(f"Dry run: valid; {job.total_bytes_processed / (1024 ** 2):.1f} MiB. "
                        "Nothing created. Pass --execute to apply.")

    if drop:
        archived = _archive_rows(client, archive_path)
        state['archived_rows'] = archived
        if source_rows == 0:
            logger.info("No V1 text left in the source -- the column is already retired.")
        elif archived is None:
            raise RuntimeError(
                f"No archive at `{archive_path}`. Dropping the column would destroy "
                f"{source_rows} evaluations that exist nowhere else. Run --archive first.")
        elif archived != source_rows:
            raise RuntimeError(
                f"Archive holds {archived} rows but the source still has {source_rows} with "
                "text. The difference is evaluations the archive does not have.")
        else:
            logger.info(f"Archive verified: {archived} rows, matching the source.")

        sql = build_drop_sql(bq_path)
        logger.info(f"--- drop ---\n{sql}\n")
        if execute:
            client.query(sql).result()
            logger.info(f"Dropped `gemini_insights` from {bq_path}.")
            state['dropped'] = True
        else:
            logger.info("Dry run: DDL is not dry-runnable; nothing executed. "
                        "Pass --execute to apply.")

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH)
    parser.add_argument("--archive_path", default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--archive", action="store_true", help="Create the archive table.")
    parser.add_argument("--drop", action="store_true",
                        help="Drop the column. Refuses without a complete archive.")
    parser.add_argument("--execute", action="store_true",
                        help="Apply. Without this, SQL and counts only.")
    args = parser.parse_args()

    if not args.archive and not args.drop:
        parser.error("choose --archive, --drop, or both")

    run_retirement(args.bq_path, args.archive_path, args.archive, args.drop, args.execute)
