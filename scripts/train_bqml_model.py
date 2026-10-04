import argparse
import datetime
import logging
from typing import Any, Callable, Optional
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

from app.core.model_features import feature_select_list, feature_source_clause
from app.core.profile_freshness import (
    PreFlightEnrichmentError,
    max_allowed_enrichment_failures,
    needs_gemini_profile,
    needs_maps_lookup,
    verify_post_enrichment_rows,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREFLIGHT_BQ_TIMEOUT_SECONDS = 60.0


def _run_preflight_query(client: bigquery.Client, sql: str, timeout: float = PREFLIGHT_BQ_TIMEOUT_SECONDS):
    job = client.query(sql)
    try:
        return job.result(timeout=timeout)
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise


def build_training_select(
    project_id: str, dataset_id: str, source_table: str, extra_predicate: str = ""
) -> str:
    """The labelled feature set the model trains on.

    Extracted so `scripts/evaluate_model.py` measures the features production
    actually uses rather than a third hand-copy of them. `extra_predicate` is
    appended to the WHERE clause, which is how the evaluation harness carves
    out its train/holdout split without restating any of this.

    The list itself comes from `app/core/model_features.py`, which the
    `ML.PREDICT` subquery in `app/services/ml_prediction.py` also builds from.
    They were two hand-maintained copies until Phase 6; see D3.
    """
    return f"""
SELECT
{feature_select_list()}
{feature_source_clause(project_id, dataset_id, source_table)}
WHERE
  (m.in_scope = TRUE OR m.in_scope IS NULL)
  AND m.user_rating IS NOT NULL
  {extra_predicate}
"""


def run_jit_preflight(
    client,
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    force_maps: bool = False,
    maps_max_age_days: Optional[int] = None,
    maps_cutoff_date: Optional[Any] = None,
    force_gemini: bool = False,
    gemini_max_age_days: Optional[int] = None,
    gemini_cutoff_date: Optional[Any] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Fill in or refresh labelled rows before training on them.

    Enforces a strict pre-flight gate:
    1. Google Maps Places regeneration runs first so Gemini sees updated ratings.
    2. Gemini `AI.GENERATE` runs second (in bounded 25-row batches with timeouts).
    3. Postcode demographics enrichment runs third.
    4. Post-enrichment verification confirms `<= 5%` of targeted rows remain
       stale or missing. Any stage error, timeout, or verification shortfall
       raises `PreFlightEnrichmentError` and prevents training on stale data.
    """
    source_table = f"{project_id}.{dataset_id}.{table_id}"
    started_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=2)

    logger.info("Executing pre-flight JIT check for labeled training examples...")
    if progress_callback:
        progress_callback("🔎 Auditing freshness of labeled training set in BigQuery...")

    # A correlated subquery, not a join: 15 normalised postcodes are duplicated
    # in the demographics table, so joining it returns those labelled rows two
    # or three times (D-31, D-32). Only "did the postcode resolve?" is read.
    check_query = f"""
        SELECT m.fhrsid, m.postcode, m.maps_lookup_at, m.gemini_profiled_at,
               m.gemini_insights_structured IS NOT NULL AS has_profile,
               (SELECT MIN(d.postcode)
                FROM `{project_id}.{dataset_id}.uk_postcode_demographics` AS d
                WHERE REPLACE(UPPER(m.postcode), ' ', '') = REPLACE(UPPER(d.postcode), ' ', '')
               ) AS d_postcode
        FROM `{source_table}` AS m
        WHERE (m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NOT NULL
    """
    try:
        results = _run_preflight_query(client, check_query)
        rows = list(results)
    except Exception as e:
        raise PreFlightEnrichmentError(
            f"Pre-flight BigQuery freshness audit failed or timed out: {e}"
        ) from e

    maps_missing = [
        str(row.fhrsid) for row in rows
        if needs_maps_lookup(
            row.maps_lookup_at,
            force=force_maps,
            max_age_days=maps_max_age_days,
            cutoff_date=maps_cutoff_date,
        )
    ]
    gemini_missing = [
        str(row.fhrsid) for row in rows
        if needs_gemini_profile(
            row.has_profile,
            row.gemini_profiled_at,
            force=force_gemini,
            max_age_days=gemini_max_age_days,
            cutoff_date=gemini_cutoff_date,
        )
    ]
    postcode_missing = [
        str(row.fhrsid) for row in rows
        if getattr(row, 'd_postcode', None) is None and getattr(row, 'postcode', None) is not None
    ]

    if maps_missing:
        maps_force_regen = bool(
            force_maps or maps_max_age_days is not None or maps_cutoff_date is not None
        )
        msg = f"🗺️ Regenerating Google Maps data for {len(maps_missing)} labeled restaurant(s)..."
        logger.info(f"JIT: Found {len(maps_missing)} labeled restaurants needing Maps data. Triggering enrichment...")
        if progress_callback:
            progress_callback(msg)
        from scripts.enrich_maps_data import enrich_restaurants_by_fhrsid
        try:
            if maps_force_regen:
                updated_maps = enrich_restaurants_by_fhrsid(
                    maps_missing, limit=len(maps_missing), force_regen=True
                )
            else:
                updated_maps = enrich_restaurants_by_fhrsid(
                    maps_missing, limit=len(maps_missing)
                )
        except PreFlightEnrichmentError:
            raise
        except Exception as e:
            raise PreFlightEnrichmentError(
                f"Maps enrichment failed before model training: {e}"
            ) from e

        if isinstance(updated_maps, int):
            allowed_maps = max_allowed_enrichment_failures(len(maps_missing))
            shortfall = len(maps_missing) - updated_maps
            if shortfall > allowed_maps:
                raise PreFlightEnrichmentError(
                    f"Maps enrichment updated only {updated_maps}/{len(maps_missing)} "
                    f"restaurants ({shortfall} unrefreshed, allowed <= {allowed_maps})."
                )

    if gemini_missing:
        logger.info(f"JIT: Found {len(gemini_missing)} labeled restaurants needing Gemini insights. Triggering enrichment...")
        if progress_callback:
            progress_callback(
                f"✨ Regenerating Gemini profiles for {len(gemini_missing)} labeled restaurant(s)..."
            )
        from app.services.bq_utils import execute_gemini_enrichment
        try:
            gemini_kwargs: dict[str, Any] = {"fhrsids": gemini_missing}
            if progress_callback is not None:
                gemini_kwargs["progress_callback"] = progress_callback
            gemini_ok = execute_gemini_enrichment(
                project_id, dataset_id, table_id, **gemini_kwargs
            )
        except PreFlightEnrichmentError:
            raise
        except Exception as e:
            raise PreFlightEnrichmentError(
                f"Gemini enrichment failed before model training: {e}"
            ) from e
        if gemini_ok is False:
            raise PreFlightEnrichmentError(
                f"Gemini enrichment failed or timed out for {len(gemini_missing)} "
                f"labeled restaurant(s); aborting model training."
            )

    if postcode_missing:
        logger.info("JIT: Found labeled restaurants missing postcode demographics. Triggering enrichment...")
        if progress_callback:
            progress_callback(
                f"📮 Enriching UK postcode demographics for {len(postcode_missing)} restaurant(s)..."
            )
        from scripts.enrich_postcode_demographics import enrich_postcodes
        try:
            enriched_postcodes = enrich_postcodes(
                project_id=project_id, dataset_id=dataset_id, master_table=table_id
            )
            if isinstance(enriched_postcodes, int) and enriched_postcodes > 0:
                logger.info(f"JIT: Enriched {enriched_postcodes} new postcodes with demographic data.")
        except PreFlightEnrichmentError:
            raise
        except Exception as e:
            raise PreFlightEnrichmentError(
                f"Postcode demographics enrichment failed before model training: {e}"
            ) from e

    if maps_missing or gemini_missing:
        if progress_callback:
            progress_callback("✅ Verifying regenerated profile freshness in BigQuery...")
        try:
            verify_results = _run_preflight_query(client, check_query)
            post_rows = list(verify_results)
        except StopIteration:
            post_rows = []
        except Exception as e:
            raise PreFlightEnrichmentError(
                f"Post-enrichment BigQuery verification failed or timed out: {e}"
            ) from e

        if post_rows:
            verify_post_enrichment_rows(
                post_rows,
                maps_targeted_fhrsids=maps_missing,
                gemini_targeted_fhrsids=gemini_missing,
                started_at=started_at,
                maps_max_age_days=maps_max_age_days,
                maps_cutoff_date=maps_cutoff_date,
                force_maps=force_maps,
                gemini_max_age_days=gemini_max_age_days,
                gemini_cutoff_date=gemini_cutoff_date,
                force_gemini=force_gemini,
            )


def train_model(
    project_id: str,
    dataset_id: str,
    table_id: str,
    model_name: str,
    dry_run: bool = False,
    run_async: bool = False,
    *,
    force_maps: bool = False,
    maps_max_age_days: Optional[int] = None,
    maps_cutoff_date: Optional[Any] = None,
    force_gemini: bool = False,
    gemini_max_age_days: Optional[int] = None,
    gemini_cutoff_date: Optional[Any] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """
    Constructs and executes a BQML model training query.

    Returns the job id of the training job, or -- on a dry run, which starts no
    job -- the number of bytes the query would process. Each mode returns the
    only answer it has: a dry run's result *is* the byte estimate, and a caller
    that asked for one cannot be handed a job id that does not exist.
    """
    client = bigquery.Client(project=project_id)

    full_model_name = f"{project_id}.{dataset_id}.{model_name}"
    source_table = f"{project_id}.{dataset_id}.{table_id}"

    # The pre-flight is skipped on a dry run, not merely made cheaper. It is
    # the half of this function that spends money, and `--dry-run` is the flag
    # someone reaches for when they are unsure what a run will do (D15).
    if dry_run:
        logger.info("Dry run: skipping the JIT enrichment pre-flight. No Places "
                    "or Gemini calls will be made, and the training SQL will be "
                    "validated against whatever the table already holds.")
    else:
        run_jit_preflight(
            client,
            project_id,
            dataset_id,
            table_id,
            force_maps=force_maps,
            maps_max_age_days=maps_max_age_days,
            maps_cutoff_date=maps_cutoff_date,
            force_gemini=force_gemini,
            gemini_max_age_days=gemini_max_age_days,
            gemini_cutoff_date=gemini_cutoff_date,
            progress_callback=progress_callback,
        )

    # We omit BusinessType as it is not present in the BigQuery schema for fsa_master.
    query = f"""
    CREATE OR REPLACE MODEL `{full_model_name}`
    OPTIONS(
      model_type='BOOSTED_TREE_REGRESSOR',
      input_label_cols=['user_rating'],
      model_registry='vertex_ai'
    ) AS
    {build_training_select(project_id, dataset_id, source_table)}
    """

    logger.info(f"Preparing BQML Training Query for {full_model_name}...")
    if dry_run:
        logger.info("Executing DRY RUN to validate query without training.")
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            query_job = client.query(query, job_config=job_config)
            logger.info("Dry run successful. Query is valid.")
            logger.info(f"This query will process {query_job.total_bytes_processed} bytes.")
            return query_job.total_bytes_processed
        except GoogleCloudError as e:
            logger.error(f"BigQuery validation failed: {e}")
            raise
    else:
        if progress_callback:
            progress_callback("🚀 Submitting BQML BOOSTED_TREE_REGRESSOR training job...")
        logger.info("Executing query. This may take 10-15 minutes for BOOSTED_TREE_REGRESSOR...")
        try:
            query_job = client.query(query)
            if run_async:
                logger.info(f"Started async model training. Job ID: {query_job.job_id}")
                return query_job.job_id

            query_job.result()  # Wait for the job to complete
            logger.info(f"Model {full_model_name} trained successfully.")
            return query_job.job_id
        except GoogleCloudError as e:
            logger.error(f"BigQuery execution failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

def training_job_status(project_id: str, dataset_id: str, job_id: str) -> dict:
    """Look up a training job started earlier by `train_model(run_async=True)`.

    `run_async` hands back a job id and returns immediately; ten to fifteen
    minutes later that job has either produced a model or failed, and nothing
    was reading the difference (D-28). This is the read.

    The job's location is the dataset's -- BigQuery runs a query where the data
    lives, and `jobs.get` needs it for anything outside the US, which this EU
    dataset is. Both calls are free job/dataset metadata reads, not queries.

    Returns `{"state": ..., "error": ...}`, where `state` is BigQuery's own
    ("PENDING", "RUNNING", "DONE") and `error` is the failure message or None.
    A finished job and a *successful* job are not the same thing: a failed
    query is `DONE` with `error_result` set.
    """
    client = bigquery.Client(project=project_id)
    location = client.get_dataset(f"{project_id}.{dataset_id}").location
    job = client.get_job(job_id, location=location)
    failure = job.error_result or {}
    return {"state": job.state, "error": failure.get("message") or None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BQML Restaurant Preference Model")
    parser.add_argument("--project_id", default="filipegracio-ai-learning", help="GCP Project ID")
    parser.add_argument("--dataset_id", default="filipegracio_fsa_restaurants", help="BigQuery Dataset ID")
    parser.add_argument("--table_id", default="fsa_master", help="Source Table ID")
    parser.add_argument("--model_name", default="restaurant_preference_model", help="Target Model Name")
    parser.add_argument("--dry-run", action="store_true", help="Validate query without executing training")
    parser.add_argument("--run_async", action="store_true", help="Run model training asynchronously")
    parser.add_argument("--force-maps", action="store_true", help="Force re-query Google Maps for all labeled rows")
    parser.add_argument("--maps-max-age-days", type=int, default=None, help="Refresh Maps lookups older than N days")
    parser.add_argument("--maps-cutoff-date", type=str, default=None, help="Refresh Maps lookups before YYYY-MM-DD")
    parser.add_argument("--force-gemini", action="store_true", help="Force regenerate Gemini profiles for all labeled rows")
    parser.add_argument("--gemini-max-age-days", type=int, default=None, help="Refresh Gemini profiles older than N days")
    parser.add_argument("--gemini-cutoff-date", type=str, default=None, help="Refresh Gemini profiles before YYYY-MM-DD")
    args = parser.parse_args()
    
    train_model(
        args.project_id,
        args.dataset_id,
        args.table_id,
        args.model_name,
        dry_run=args.dry_run,
        run_async=args.run_async,
        force_maps=args.force_maps,
        maps_max_age_days=args.maps_max_age_days,
        maps_cutoff_date=args.maps_cutoff_date,
        force_gemini=args.force_gemini,
        gemini_max_age_days=args.gemini_max_age_days,
        gemini_cutoff_date=args.gemini_cutoff_date,
    )
