"""Read-only reconnaissance of `fsa_master` before the pillar-column migration.

Nothing in this repo records what `gemini_insights_structured` actually
contains. Four mutually inconsistent key conventions exist across the prompt,
the training SQL, the UI parser and the recorded ADK output, and the 17-column
schema in the track spec was designed against the prompt text rather than
against observed rows. This script settles that empirically.

Every query here is a SELECT. The only statement that writes is the snapshot,
which is behind its own flag.

    python -m scripts.recon_pipeline_state                 # print SQL + cost, run nothing
    python -m scripts.recon_pipeline_state --execute       # run the read-only census
    python -m scripts.recon_pipeline_state --snapshot --execute   # snapshot first, then census

Dry run is the default, which is the opposite of the other `scripts/migrate_*`
here: those default to executing. Recon is the step whose output decides the
schema, so it is worth seeing the SQL before it runs.
"""
import argparse
import json
import logging
import pathlib

from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
DEFAULT_SNAPSHOT_SUFFIX = "backup_20260923"
FIXTURE_DIR = pathlib.Path("tests/fixtures/gemini_profiles")

# On-demand analysis, London (europe-west2): $6.25 per TiB. Held here as a
# constant so the number in the log is auditable rather than folded into a
# magic multiplication.
USD_PER_TIB = 6.25
USD_TO_GBP = 0.79

# The unwrap `train_bqml_model.py` and `ml_prediction.py` both apply before
# every JSON read. Recon must measure the same string the pipeline parses, or
# it answers a question nobody asked. Kept as a plain string: the production
# copies live inside `.format()` templates and so carry doubled braces.
JSON_UNWRAP_REGEX = "r'(?s)[{].*[}]'"

# The six paths the production feature extraction reads today. Verified against
# the SQL by scripts/test_recon_pipeline_state.py, so this list cannot silently
# drift from what the model is actually fed.
PRODUCTION_FEATURE_PATHS = [
    '$.1_value_and_volume_rating',
    '$.2_demographic_community_score',
    '$.3_linguistic_signal_score',
    '$.4_geographic_precision_specificity_level',
    '$.5_culinary_uncompromisingness_score',
    '$.match_score',
]

# The same six as the prompt's `### EXAMPLE OUTPUT` nests them, plus pillar 6,
# which no feature reads but `migrate_to_in_scope_workflow.py` gated on.
PROMPT_NESTED_PATHS = [
    '$.1_value_and_volume.rating',
    '$.2_demographic_community.score',
    '$.3_linguistic_signal.score',
    '$.4_geographic_precision.specificity_level',
    '$.5_culinary_uncompromisingness.score',
    '$.6_establishment_integrity.is_sit_down_restaurant',
    '$.match_score',
]

FLAT_SIT_DOWN_PATH = '$.6_establishment_integrity_is_sit_down_restaurant'


def _unwrapped(column: str = "gemini_insights_structured") -> str:
    return f"REGEXP_EXTRACT({column}, {JSON_UNWRAP_REGEX})"


def _path_census_query(bq_path: str, paths, label: str) -> str:
    """Per-path non-null and distinct-value counts.

    A path that resolves on zero rows is a dead feature. A path that resolves
    everywhere but has one distinct value is a constant, which trains just as
    badly and is invisible in a NULL count.
    """
    unwrapped = _unwrapped()
    selects = []
    for i, path in enumerate(paths):
        expr = f"JSON_EXTRACT_SCALAR(u, '{path}')"
        selects.append(f"  COUNTIF({expr} IS NOT NULL) AS p{i}_nonnull")
        selects.append(f"  COUNT(DISTINCT {expr}) AS p{i}_distinct")
    body = ",\n".join(selects)
    return f"""-- {label}
SELECT
{body}
FROM (
  SELECT {unwrapped} AS u
  FROM `{bq_path}`
  WHERE gemini_insights_structured IS NOT NULL
)"""


def build_queries(bq_path: str) -> dict:
    """The read-only census. Keys become the section names in the report."""
    unwrapped = _unwrapped()

    queries = {}

    queries['sizing'] = f"""-- Row counts that size every later phase
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(gemini_insights_structured IS NOT NULL) AS profiled_rows,
  COUNTIF(gemini_insights IS NOT NULL) AS legacy_v1_populated,
  COUNTIF(user_rating IS NOT NULL) AS labelled_rows,
  COUNTIF(maps_rating = -1) AS maps_miss_sentinel,
  COUNTIF(maps_rating > 0) AS maps_hit,
  COUNTIF(maps_rating IS NULL) AS maps_never_looked_up,
  COUNTIF(latitude IS NOT NULL AND longitude IS NOT NULL) AS rows_with_coords,
  COUNTIF(in_scope IS TRUE) AS in_scope_true,
  COUNTIF(in_scope IS FALSE) AS in_scope_false,
  COUNTIF(in_scope IS NULL) AS in_scope_null,
  COUNTIF(predicted_user_rating IS NOT NULL) AS predicted_rows
FROM `{bq_path}`"""

    # D1 is already fixed, but the empirical confirmation belongs in the log:
    # if legacy_v1_populated is 0 then the column the old guard tested could
    # never have indicated a cached profile.
    queries['parseability'] = f"""-- Can the stored text even be read as JSON?
SELECT
  COUNT(*) AS profiled_rows,
  COUNTIF({unwrapped} IS NULL) AS no_json_object_found,
  COUNTIF(SAFE.PARSE_JSON({unwrapped}) IS NULL) AS unparseable_after_unwrap,
  COUNTIF(gemini_insights_structured LIKE '%```%') AS wrapped_in_markdown
FROM `{bq_path}`
WHERE gemini_insights_structured IS NOT NULL"""

    queries['key_census'] = f"""-- Every key path to depth 2, with how many rows carry it.
-- Settles which of the four conventions the column actually uses.
SELECT
  key_path,
  COUNT(*) AS rows_with_key
FROM `{bq_path}`,
  UNNEST(JSON_KEYS(SAFE.PARSE_JSON({unwrapped}), 2)) AS key_path
WHERE gemini_insights_structured IS NOT NULL
GROUP BY key_path
ORDER BY rows_with_key DESC, key_path"""

    queries['shape_stability'] = f"""-- Distinct top-level key sets. One row back = the shape is stable
-- and a fixed path set is safe. Many rows = D14 is live and paths alone
-- will not hold.
WITH shapes AS (
  SELECT
    TO_JSON_STRING(ARRAY(
      SELECT k FROM UNNEST(JSON_KEYS(SAFE.PARSE_JSON({unwrapped}), 1)) AS k ORDER BY k
    )) AS shape
  FROM `{bq_path}`
  WHERE gemini_insights_structured IS NOT NULL
)
SELECT shape, COUNT(*) AS n_rows
FROM shapes
GROUP BY shape
ORDER BY n_rows DESC
LIMIT 25"""

    queries['production_paths'] = _path_census_query(
        bq_path, PRODUCTION_FEATURE_PATHS,
        'D2: the flat paths train_bqml_model.py and ml_prediction.py read today'
    )

    queries['prompt_paths'] = _path_census_query(
        bq_path, PROMPT_NESTED_PATHS,
        "D2: the nested paths the prompt's EXAMPLE OUTPUT promises"
    )

    queries['in_scope_derivation'] = f"""-- D13: migrate_to_in_scope_workflow.py gated on the FLAT sit-down path.
-- If that never resolved, those branches never fired and in_scope was
-- assigned from maps_types alone. The last two columns size the damage.
SELECT
  COUNTIF(JSON_EXTRACT_SCALAR(u, '{FLAT_SIT_DOWN_PATH}') IS NOT NULL) AS flat_path_resolves,
  COUNTIF(JSON_EXTRACT_SCALAR(u, '$.6_establishment_integrity.is_sit_down_restaurant') IS NOT NULL) AS nested_path_resolves,
  COUNTIF(in_scope IS FALSE
    AND LOWER(JSON_EXTRACT_SCALAR(u, '$.6_establishment_integrity.is_sit_down_restaurant')) = 'true'
  ) AS out_of_scope_but_is_sit_down,
  COUNTIF(in_scope IS TRUE
    AND LOWER(JSON_EXTRACT_SCALAR(u, '$.6_establishment_integrity.is_sit_down_restaurant')) = 'false'
  ) AS in_scope_but_not_sit_down
FROM (
  SELECT in_scope, {unwrapped} AS u
  FROM `{bq_path}`
  WHERE gemini_insights_structured IS NOT NULL
)"""

    queries['label_distribution'] = f"""-- The training set. Phase 2's held-out split has to be sized against this.
SELECT user_rating, COUNT(*) AS n_rows
FROM `{bq_path}`
WHERE user_rating IS NOT NULL
GROUP BY user_rating
ORDER BY user_rating"""

    return queries


def fixture_query(bq_path: str, limit: int) -> str:
    """Real payloads to pin the Phase 3 contract test against.

    Ordered by fhrsid rather than sampled at random so a re-run captures the
    same rows and the fixtures stay reviewable in a diff.
    """
    return f"""SELECT fhrsid, gemini_insights_structured
FROM `{bq_path}`
WHERE gemini_insights_structured IS NOT NULL
ORDER BY fhrsid
LIMIT {limit}"""


def snapshot_query(bq_path: str, snapshot_table: str) -> str:
    """The restore point for every backfill in this track.

    `fsa_master` holds the only copy of the hand-entered `user_rating` labels
    and nothing in the repo backs them up.
    """
    project_id, dataset_id, _ = bq_path.split(".")
    return f"""CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{snapshot_table}`
AS SELECT * FROM `{bq_path}`"""


def estimate_cost(client: bigquery.Client, sql: str) -> int:
    """Bytes this query would scan, via a BigQuery dry run. Costs nothing."""
    config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    return client.query(sql, job_config=config).total_bytes_processed


def _format_cost(total_bytes: int) -> str:
    tib = total_bytes / (1024 ** 4)
    gbp = tib * USD_PER_TIB * USD_TO_GBP
    return f"{total_bytes / (1024 ** 2):.1f} MiB scanned, about £{gbp:.4f}"


def run_recon(bq_path: str = DEFAULT_BQ_PATH, execute: bool = False,
              snapshot: bool = False, snapshot_table: str = "",
              fixture_limit: int = 8) -> dict:
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    snapshot_table = snapshot_table or f"{bq_path.split('.')[2]}_{DEFAULT_SNAPSHOT_SUFFIX}"

    results = {}

    if snapshot:
        sql = snapshot_query(bq_path, snapshot_table)
        logger.info(f"SNAPSHOT (this one writes):\n{sql}\n")
        if execute:
            client.query(sql).result()
            logger.info(f"Snapshot written to {snapshot_table}.")
        else:
            logger.info("[DRY RUN] Snapshot not created.")

    queries = build_queries(bq_path)
    total_bytes = 0

    for name, sql in queries.items():
        logger.info(f"--- {name} ---\n{sql}\n")
        try:
            scanned = estimate_cost(client, sql)
            total_bytes += scanned
            logger.info(f"[{name}] {_format_cost(scanned)}")
        except Exception as e:
            # A dry run failing is itself a finding -- usually a column or a
            # JSON function that does not exist in this region.
            logger.error(f"[{name}] dry run failed: {e}")
            results[name] = {'error': str(e)}
            continue

        if not execute:
            continue

        try:
            rows = [dict(r) for r in client.query(sql).result()]
            results[name] = rows
            logger.info(f"[{name}] {len(rows)} row(s) returned")
        except Exception as e:
            logger.error(f"[{name}] failed: {e}")
            results[name] = {'error': str(e)}

    logger.info(f"TOTAL for the census: {_format_cost(total_bytes)}")

    if execute:
        _capture_fixtures(client, bq_path, fixture_limit)

    return results


def _capture_fixtures(client: bigquery.Client, bq_path: str, limit: int) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(client.query(fixture_query(bq_path, limit)).result())
    for row in rows:
        path = FIXTURE_DIR / f"{row.fhrsid}.json"
        path.write_text(row.gemini_insights_structured or "")
    logger.info(f"Wrote {len(rows)} payload fixture(s) to {FIXTURE_DIR}/")


def write_report(results: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Report written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH)
    parser.add_argument("--execute", action="store_true",
                        help="Actually run the queries. Without this, SQL and cost are printed only.")
    parser.add_argument("--snapshot", action="store_true",
                        help="Also copy the table to a dated backup before reading. This writes.")
    parser.add_argument("--snapshot_table", default="",
                        help=f"Snapshot table name. Defaults to <table>_{DEFAULT_SNAPSHOT_SUFFIX}.")
    parser.add_argument("--fixture_limit", type=int, default=8)
    parser.add_argument("--report", default="", help="Write results as JSON to this path.")
    args = parser.parse_args()

    out = run_recon(
        bq_path=args.bq_path, execute=args.execute, snapshot=args.snapshot,
        snapshot_table=args.snapshot_table, fixture_limit=args.fixture_limit,
    )
    if args.report and out:
        write_report(out, args.report)
