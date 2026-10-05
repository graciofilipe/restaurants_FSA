import argparse
import datetime
import logging
from typing import Any, Callable, Optional
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

from app.core.model_features import (
    counterweight_linear_replication_sql,
    counterweight_linear_where_clause,
    feature_select_list,
    feature_source_clause,
    normalized_brand_key_sql,
    stage2_training_where_clause,
)
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
LINEAR_MODEL_SUFFIX = "_linear"


def companion_linear_model_name(model_name: str) -> str:
    """Return the companion Course 2b linear regression model name for a given
    primary model name."""
    if model_name.endswith(LINEAR_MODEL_SUFFIX):
        return model_name
    return f"{model_name}{LINEAR_MODEL_SUFFIX}"


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
    project_id: str,
    dataset_id: str,
    source_table: str,
    extra_predicate: str = "",
    model_family: str = "boosted_tree",
) -> str:
    """The labelled feature set the model trains on.

    Extracted so `scripts/evaluate_model.py` measures the features production
    actually uses rather than a third hand-copy of them. `extra_predicate` is
    appended to the WHERE clause, which is how the evaluation harness carves
    out its train/holdout split without restating any of this.

    Hybrid Ensemble routing:
    * `boosted_tree` (Course 1b — Location-Blind Stage-2 Tree):
      Selects `feature_select_list(include_location=False)` (omitting
      `localauthorityname` and `imd_rank` for 0.000 borough bias), filters
      exclusively to Stage-2 sit-down `RESTAURANT_DINING` candidates with `< 5`
      branches (`stage2_training_where_clause`), and weights visited ground
      truth (`rating_source = 'visited'`) 2x relative to desk/NULL triage labels
      (1x).
    * `linear_reg` (Course 2b — 4:2:1 Counter-Weighted All-Scope Linear):
      Selects `feature_select_list(include_location=True)` (retaining
      `localauthorityname`, `imd_rank`, and Stage-1 regime columns), trains on
      all `in_scope` labeled rows (`counterweight_linear_where_clause`), and
      applies `4:2:1` integer replication (`counterweight_linear_replication_sql`)
      so non-plausible local rows anchor borough intercepts without stealing
      tree splits.
    """
    if model_family == "linear_reg":
        return f"""
SELECT
{feature_select_list(include_location=True)}
{feature_source_clause(project_id, dataset_id, source_table)}
CROSS JOIN UNNEST(GENERATE_ARRAY(1, {counterweight_linear_replication_sql('m', 'b')})) AS _rep
WHERE
  {counterweight_linear_where_clause('m')}
  {extra_predicate}
"""
    if model_family != "boosted_tree":
        raise ValueError(f"Unsupported model_family: {model_family!r}")
    return f"""
SELECT
{feature_select_list(include_location=False)}
{feature_source_clause(project_id, dataset_id, source_table)}
CROSS JOIN UNNEST(GENERATE_ARRAY(1, IF(m.rating_source = 'visited', 2, 1))) AS _rep
WHERE
  {stage2_training_where_clause('m', 'b')}
  {extra_predicate}
"""


def build_create_model_sql(
    project_id: str,
    dataset_id: str,
    source_table: str,
    full_model_name: str,
    model_family: str = "boosted_tree",
    extra_predicate: str = "",
    register_vertex: bool = True,
) -> str:
    """Construct the `CREATE OR REPLACE MODEL` DDL for Stage-2 / Hybrid training."""
    registry_opt = ",\n      model_registry='vertex_ai'" if register_vertex else ""
    if model_family == "linear_reg":
        options = (
            "model_type='LINEAR_REG',\n"
            "      input_label_cols=['user_rating'],\n"
            "      l2_reg=1.0,\n"
            f"      data_split_method='NO_SPLIT'{registry_opt}"
        )
    elif model_family == "boosted_tree":
        options = (
            "model_type='BOOSTED_TREE_REGRESSOR',\n"
            "      input_label_cols=['user_rating'],\n"
            "      max_tree_depth=3,\n"
            "      min_tree_child_weight=6,\n"
            "      learn_rate=0.1,\n"
            "      subsample=0.8,\n"
            "      colsample_bytree=0.8,\n"
            "      l2_reg=1.0,\n"
            "      num_parallel_tree=1,\n"
            "      max_iterations=25,\n"
            f"      data_split_method='NO_SPLIT'{registry_opt}"
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family!r}")

    return f"""
    CREATE OR REPLACE MODEL `{full_model_name}`
    OPTIONS(
      {options}
    ) AS
    {build_training_select(project_id, dataset_id, source_table, extra_predicate=extra_predicate, model_family=model_family)}
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
    refresh_missing_stage2_pillars: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Fill in or refresh labelled Stage-2 candidate rows before training on them.

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

    logger.info("Executing pre-flight JIT check for labeled Stage-2 training examples...")
    if progress_callback:
        progress_callback("🔎 Auditing freshness of labeled Stage-2 training set in BigQuery...")

    # A correlated subquery, not a join: 15 normalised postcodes are duplicated
    # in the demographics table, so joining it returns those labelled rows two
    # or three times (D-31, D-32). Only "did the postcode resolve?" is read.
    # Scoped to Stage-2 sit-down RESTAURANT_DINING candidates (plus unprofiled
    # labelled rows whose pillar columns are NULL until Gemini profiles them).
    check_query = f"""
        SELECT m.fhrsid, m.postcode, m.maps_lookup_at, m.gemini_profiled_at,
               m.gemini_insights_structured IS NOT NULL AS has_profile,
               m.pillar_dining_pace IS NOT NULL AS has_stage2_pillars,
               (SELECT MIN(d.postcode)
                FROM `{project_id}.{dataset_id}.uk_postcode_demographics` AS d
                WHERE REPLACE(UPPER(m.postcode), ' ', '') = REPLACE(UPPER(d.postcode), ' ', '')
               ) AS d_postcode
        FROM `{source_table}` AS m
        WHERE (m.in_scope = TRUE OR m.in_scope IS NULL)
          AND m.user_rating IS NOT NULL
          AND (m.pillar_is_sit_down IS TRUE OR m.pillar_is_sit_down IS NULL)
          AND (m.pillar_establishment_type = 'RESTAURANT_DINING' OR m.pillar_establishment_type IS NULL)
          AND (
            SELECT COUNT(DISTINCT b_src.fhrsid)
            FROM `{source_table}` AS b_src
            WHERE {normalized_brand_key_sql('b_src')} = {normalized_brand_key_sql('m')}
          ) < 5
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
        ) or (
            refresh_missing_stage2_pillars
            and not getattr(row, "has_stage2_pillars", False)
        )
    ]
    postcode_missing = [
        str(row.fhrsid) for row in rows
        if getattr(row, 'd_postcode', None) is None and getattr(row, 'postcode', None) is not None
    ]

    if progress_callback:
        gemini_batches = (len(gemini_missing) + 24) // 25
        progress_callback(
            f"📋 Audit complete ({len(rows)} labeled restaurant(s)): "
            f"{len(maps_missing)} Maps lookups · "
            f"{len(gemini_missing)} Gemini profiles ({gemini_batches} batch(es) of 25) · "
            f"{len(postcode_missing)} Postcode lookups."
        )

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
            maps_kwargs: dict[str, Any] = {"limit": len(maps_missing)}
            if maps_force_regen:
                maps_kwargs["force_regen"] = True
            if progress_callback is not None:
                maps_kwargs["progress_callback"] = progress_callback
            updated_maps = enrich_restaurants_by_fhrsid(
                maps_missing, **maps_kwargs
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
    elif progress_callback:
        progress_callback(
            f"🗺️ Google Maps data: all {len(rows)} labeled restaurant(s) are already fresh (0 to regenerate)."
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
    elif progress_callback:
        progress_callback(
            f"✨ Gemini profiles: all {len(rows)} labeled restaurant(s) are already fresh (0 to regenerate)."
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
                force_gemini=force_gemini or refresh_missing_stage2_pillars,
            )


def train_model(
    project_id: str,
    dataset_id: str,
    table_id: str,
    model_name: str,
    dry_run: bool = False,
    run_async: bool = False,
    *,
    model_family: str = "boosted_tree",
    train_companion_linear: bool = True,
    force_maps: bool = False,
    maps_max_age_days: Optional[int] = None,
    maps_cutoff_date: Optional[Any] = None,
    force_gemini: bool = False,
    gemini_max_age_days: Optional[int] = None,
    gemini_cutoff_date: Optional[Any] = None,
    refresh_missing_stage2_pillars: bool = False,
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
            refresh_missing_stage2_pillars=refresh_missing_stage2_pillars,
            progress_callback=progress_callback,
        )

    tree_query = build_create_model_sql(
        project_id,
        dataset_id,
        source_table,
        full_model_name,
        model_family=model_family,
    )

    should_train_companion = train_companion_linear and model_family == "boosted_tree"
    if should_train_companion:
        linear_model_name = companion_linear_model_name(model_name)
        full_linear_model_name = f"{project_id}.{dataset_id}.{linear_model_name}"
        linear_query = build_create_model_sql(
            project_id,
            dataset_id,
            source_table,
            full_linear_model_name,
            model_family="linear_reg",
            register_vertex=False,
        )
        query = f"{linear_query.strip()};\n{tree_query.strip()}"
    else:
        query = tree_query

    logger.info(f"Preparing BQML Training Query for {full_model_name} (family={model_family})...")
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
            progress_callback(f"🚀 Submitting BQML {model_family} training job...")
        logger.info(f"Executing query for {model_family}...")
        try:
            query_job = client.query(query)
            if run_async:
                logger.info(f"Started async model training. Job ID: {query_job.job_id}")
                return query_job.job_id

            query_job.result()  # Wait for the job to complete
            logger.info(f"Model {full_model_name} trained successfully.")
            from app.services.ml_prediction import rescore_all_in_scope_predictions
            rescore_all_in_scope_predictions(
                project_id=project_id,
                dataset_id=dataset_id,
                table_id=table_id,
                model_name=model_name,
                client=client,
                progress_callback=progress_callback,
            )
            return query_job.job_id
        except GoogleCloudError as e:
            logger.error(f"BigQuery execution failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

def training_job_status(project_id: str, dataset_id: str, job_id: str) -> dict:
    """Look up a training job started earlier by `train_model(run_async=True)`."""
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
    parser.add_argument(
        "--model-family",
        choices=["boosted_tree", "linear_reg"],
        default="boosted_tree",
        help="BQML model family to train (boosted_tree or linear_reg)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate query without executing training")
    parser.add_argument("--run_async", action="store_true", help="Run model training asynchronously")
    parser.add_argument("--force-maps", action="store_true", help="Force re-query Google Maps for all labeled rows")
    parser.add_argument("--maps-max-age-days", type=int, default=None, help="Refresh Maps lookups older than N days")
    parser.add_argument("--maps-cutoff-date", type=str, default=None, help="Refresh Maps lookups before YYYY-MM-DD")
    parser.add_argument("--force-gemini", action="store_true", help="Force regenerate Gemini profiles for all labeled rows")
    parser.add_argument("--gemini-max-age-days", type=int, default=None, help="Refresh Gemini profiles older than N days")
    parser.add_argument("--gemini-cutoff-date", type=str, default=None, help="Refresh Gemini profiles before YYYY-MM-DD")
    parser.add_argument(
        "--refresh-missing-stage2-pillars",
        action="store_true",
        help="Regenerate Gemini profiles for Stage-2 labeled rows missing Pillar 7 discriminators",
    )
    args = parser.parse_args()

    train_model(
        args.project_id,
        args.dataset_id,
        args.table_id,
        args.model_name,
        dry_run=args.dry_run,
        run_async=args.run_async,
        model_family=args.model_family,
        force_maps=args.force_maps,
        maps_max_age_days=args.maps_max_age_days,
        maps_cutoff_date=args.maps_cutoff_date,
        force_gemini=args.force_gemini,
        gemini_max_age_days=args.gemini_max_age_days,
        gemini_cutoff_date=args.gemini_cutoff_date,
        refresh_missing_stage2_pillars=args.refresh_missing_stage2_pillars,
    )
