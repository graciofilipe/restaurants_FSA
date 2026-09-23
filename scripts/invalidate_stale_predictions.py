"""Clear predictions made by a model that has since been replaced.

The Phase 6 retrain changed the model's input schema -- five pillar features
that were pinned to zero became real, and two more were added. Every
`predicted_user_rating` written before that retrain came out of the old model,
but nothing in the table says so: `predicted_at` records *when* a row was
scored, not *what* scored it. The priority heuristic then reads a recent
timestamp as "freshly scored" and parks the row at the bottom of the queue,
so a stale prediction is self-perpetuating.

Nulling the pair puts those rows back in the unscored tier, which is where a
row scored by a retired model belongs.

    python -m scripts.invalidate_stale_predictions              # validate, run nothing
    python -m scripts.invalidate_stale_predictions --execute    # production

Dry run is the default. The cutoff is read from the served model's own
`last_modified_time` rather than typed in, because a hand-entered date that is
slightly too late silently leaves stale rows behind, and one that is slightly
too early clears good ones.

**Cost.** The `UPDATE` itself is a few MiB. What follows it is not free: a
cleared row re-enters the queue, and re-scoring any row without a Gemini
profile bills a fresh `AI.GENERATE`. The dry run reports that count separately
so the bill is visible before the write, not after.
"""
import argparse
import datetime
import logging

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
DEFAULT_MODEL_NAME = "restaurant_preference_model"


def read_model_trained_at(client, project_id: str, dataset_id: str,
                          model_name: str) -> datetime.datetime:
    """When the served model was last trained, from the model's own metadata.

    `created`, not `modified`: `CREATE OR REPLACE MODEL` resets the creation
    time, so for a BQML model it *is* the training time, while `modified` also
    moves for metadata-only edits like a description or a label -- which would
    silently widen the cutoff and clear good predictions.

    Read through `get_model` rather than `INFORMATION_SCHEMA.MODELS`: the
    view is region-scoped and a dataset-qualified query for it resolves against
    the job's location, which fails here even though the dataset is in EU.
    """
    model = client.get_model(f"{project_id}.{dataset_id}.{model_name}")
    return model.created


def build_impact_query(bq_path: str, trained_at: datetime.datetime) -> str:
    """What the write would touch, split by what re-scoring each row costs."""
    return f"""SELECT
  COUNTIF(predicted_at < TIMESTAMP('{trained_at}')) AS stale_predictions,
  COUNTIF(predicted_at >= TIMESTAMP('{trained_at}')) AS current_predictions,
  COUNTIF(predicted_at < TIMESTAMP('{trained_at}')
          AND gemini_profiled_at IS NOT NULL) AS stale_but_profiled,
  COUNTIF(predicted_at < TIMESTAMP('{trained_at}')
          AND gemini_profiled_at IS NULL) AS stale_and_unprofiled,
  COUNTIF(predicted_at < TIMESTAMP('{trained_at}')
          AND user_rating IS NOT NULL) AS stale_but_labelled
FROM `{bq_path}`"""


def build_invalidate_sql(bq_path: str, trained_at: datetime.datetime) -> str:
    """Null the score and its stamp for rows the retired model scored.

    Strictly `<` the cutoff. `predicted_at IS NOT NULL` would also catch rows
    scored by the current model, and `OR predicted_at IS NULL` would rewrite
    rows that were never scored at all -- both turn a targeted invalidation
    into a column-wide one.
    """
    return f"""UPDATE `{bq_path}`
SET predicted_user_rating = NULL, predicted_at = NULL
WHERE predicted_at < TIMESTAMP('{trained_at}')"""


def run_invalidation(bq_path: str = DEFAULT_BQ_PATH,
                     model_name: str = DEFAULT_MODEL_NAME,
                     execute: bool = False) -> dict:
    """Report the impact, then clear the stale predictions if asked to."""
    project_id, dataset_id, _ = bq_path.split(".")
    client = bigquery.Client(project=project_id)

    try:
        trained_at = read_model_trained_at(client, project_id, dataset_id, model_name)
    except NotFound as exc:
        raise RuntimeError(
            f"No model named '{model_name}' in {project_id}.{dataset_id}. Without its "
            "training time there is no cutoff, and every prediction would be cleared."
        ) from exc
    logger.info(f"Served model '{model_name}' last trained at {trained_at}.")

    impact_sql = build_impact_query(bq_path, trained_at)
    logger.info(f"--- impact ---\n{impact_sql}\n")
    impact = dict(list(client.query(impact_sql).result())[0].items())
    for key, value in impact.items():
        logger.info(f"  {key:22} {value}")
    logger.info(
        f"  Re-scoring the {impact.get('stale_and_unprofiled', 0)} unprofiled rows would "
        "bill a Gemini call each; the rest cost ML.PREDICT only.")

    sql = build_invalidate_sql(bq_path, trained_at)
    logger.info(f"--- invalidate ---\n{sql}\n")

    if not execute:
        job = client.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False))
        logger.info(f"Dry run: valid; {job.total_bytes_processed / (1024 ** 2):.1f} MiB. "
                    "Nothing written. Pass --execute to apply.")
        return impact

    job = client.query(sql)
    job.result()
    logger.info(f"Cleared {job.num_dml_affected_rows} stale predictions.")
    impact['rows_cleared'] = job.num_dml_affected_rows
    return impact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--execute", action="store_true",
                        help="Apply the UPDATE. Without this, SQL and impact only.")
    args = parser.parse_args()

    run_invalidation(args.bq_path, args.model_name, args.execute)
