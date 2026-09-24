"""Re-score the rows that already hold everything a prediction needs.

The mirror image of `invalidate_stale_predictions.py`. That script cleared
1,065 predictions written by a model the Phase 6 retrain had replaced, which
was right; nothing put any back. `predicted_user_rating` is now NULL on all
11,268 rows, so the staleness component of `calculate_restaurant_priority`
scores a constant 100 everywhere and has stopped discriminating between rows
at all. The triage queue is ordered by the other three components alone.

    python -m scripts.backfill_predictions              # validate, score nothing
    python -m scripts.backfill_predictions --execute    # production

**Cost.** Deliberately £0 beyond `ML.PREDICT`. The candidates are exactly the
rows for which `generate_predictions`' three enrichment branches are empty --
profile present and fresh, Maps looked up, demographics joined -- so scoring
them buys nothing new. That guarantee is the whole script, and a wrong one
bills Places and `AI.GENERATE` across thousands of rows, so it is not taken on
trust from the candidate query: every chunk is re-checked at run time through
`split_enrichment_targets`, the same function production uses, and the run
aborts rather than spending. The check runs on a dry run too, because
discovering the bill after `--execute` is the wrong order for that discovery.

Rows without a profile are *not* candidates and are not scored here. They cost
a Gemini call each and belong to the deferred sweep, priced in the cost ledger.
"""
import argparse
import logging

from google.cloud import bigquery

from app.core.profile_freshness import GEMINI_PROFILE_MAX_AGE_DAYS
from app.services.ml_prediction import (
    build_find_query,
    generate_predictions,
    split_enrichment_targets,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
DEFAULT_MODEL_NAME = "restaurant_preference_model"
DEFAULT_CHUNK_SIZE = 250


def build_candidate_query(bq_path: str) -> str:
    """Unscored rows that need nothing bought for them.

    Each clause is the SQL of one branch of `split_enrichment_targets`:
    `maps_lookup_at IS NOT NULL` is the Maps branch, the profile pair is the
    Gemini branch (including its freshness threshold, imported rather than
    typed, so the two cannot drift), and the `EXISTS` is the demographics
    branch. `postcode IS NULL` is a candidate rather than an exclusion: a row
    with no postcode cannot be looked up, so nothing will be bought for it.

    `EXISTS` and not a `LEFT JOIN`: the demographics table is keyed on a
    normalised postcode, and a duplicate there would fan a joined row out into
    several, inflating the count the dry run reports.
    """
    project_id, dataset_id, _ = bq_path.split(".")
    return f"""SELECT m.fhrsid
FROM `{bq_path}` AS m
WHERE m.predicted_user_rating IS NULL
  AND m.gemini_insights_structured IS NOT NULL
  AND m.gemini_profiled_at IS NOT NULL
  AND m.gemini_profiled_at > TIMESTAMP_SUB(
        CURRENT_TIMESTAMP(), INTERVAL {GEMINI_PROFILE_MAX_AGE_DAYS} DAY)
  AND m.maps_lookup_at IS NOT NULL
  AND (m.postcode IS NULL OR EXISTS (
        SELECT 1 FROM `{project_id}.{dataset_id}.uk_postcode_demographics` AS d
        WHERE REPLACE(UPPER(m.postcode), ' ', '') = REPLACE(UPPER(d.postcode), ' ', '')))
ORDER BY m.fhrsid"""


def verify_chunk_is_free(client, project_id: str, dataset_id: str, table_id: str,
                         fhrsids: list) -> int:
    """Re-derive what this chunk needs, and raise if the answer is not "nothing".

    Runs the production find query and the production split, so this cannot
    quietly diverge from what `generate_predictions` will do a moment later
    with the same ids. Returns the number of rows confirmed free.
    """
    query = build_find_query(project_id, dataset_id, table_id, target_fhrsids=fhrsids)
    rows = list(client.query(query).result())
    split = split_enrichment_targets(rows)

    needed = {step: split[step] for step in ('maps', 'gemini', 'postcodes') if split[step]}
    if needed:
        summary = "; ".join(f"{step}: {len(ids)} rows (e.g. {ids[0]})"
                            for step, ids in needed.items())
        raise RuntimeError(
            f"Refusing to run: {summary}. These rows would bill Places or AI.GENERATE, "
            "and this backfill exists to be free. The candidate query and the run-time "
            "check disagree, which means one of them is wrong -- fix that before scoring.")

    if len(rows) != len(fhrsids):
        # Either direction is a problem worth naming. Short means ids vanished
        # between the two queries; long means the find query is returning a row
        # more than once, which is how the demographics fan-out was found.
        logger.warning(f"  find query returned {len(rows)} rows for {len(fhrsids)} ids "
                       f"({'short' if len(rows) < len(fhrsids) else 'duplicated'}).")
    return len(rows)


def run_backfill(bq_path: str = DEFAULT_BQ_PATH,
                 model_name: str = DEFAULT_MODEL_NAME,
                 execute: bool = False,
                 chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    """Verify the whole batch is free, then score it in chunks."""
    project_id, dataset_id, table_id = bq_path.split(".")
    client = bigquery.Client(project=project_id)

    candidate_sql = build_candidate_query(bq_path)
    logger.info(f"--- candidates ---\n{candidate_sql}\n")
    fhrsids = [str(row.fhrsid) for row in client.query(candidate_sql).result()]
    chunks = [fhrsids[i:i + chunk_size] for i in range(0, len(fhrsids), chunk_size)]
    result = {'candidates': len(fhrsids), 'chunks': len(chunks), 'verified': 0, 'scored': 0}

    if not fhrsids:
        logger.info("No unscored rows are fully enriched. Nothing to back-fill.")
        return result

    logger.info(f"{len(fhrsids)} unscored rows need no enrichment; "
                f"{len(chunks)} chunks of up to {chunk_size}.")

    # Every chunk is checked before any chunk is scored. Verifying inside the
    # scoring loop would bill the clean chunks and then abort on the first
    # dirty one, which is a worse outcome than either running or refusing.
    for index, chunk in enumerate(chunks, start=1):
        result['verified'] += verify_chunk_is_free(client, project_id, dataset_id,
                                                   table_id, chunk)
        logger.info(f"  chunk {index}/{len(chunks)}: {len(chunk)} rows, nothing to buy.")

    if not execute:
        logger.info(f"Dry run: {result['verified']} rows verified free of Places and "
                    "Gemini calls. Nothing scored. Pass --execute to apply.")
        return result

    for index, chunk in enumerate(chunks, start=1):
        logger.info(f"Scoring chunk {index}/{len(chunks)} ({len(chunk)} rows)...")
        success, message = generate_predictions(
            project_id=project_id, dataset_id=dataset_id, table_id=table_id,
            model_name=model_name, target_fhrsids=chunk,
            force_maps=False, force_gemini=False)
        if not success:
            raise RuntimeError(
                f"Chunk {index}/{len(chunks)} failed after {result['scored']} rows "
                f"were scored: {message}")
        result['scored'] += len(chunk)
        logger.info(f"  {message}")

    logger.info(f"Back-filled {result['scored']} predictions.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--execute", action="store_true",
                        help="Score the candidates. Without this, verification only.")
    args = parser.parse_args()

    run_backfill(args.bq_path, args.model_name, args.execute, args.chunk_size)
