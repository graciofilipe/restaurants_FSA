import logging
from google.cloud import bigquery
from typing import Any, Callable, List, Optional, Tuple
from scripts.enrich_maps_data import enrich_restaurants_by_fhrsid
from app.core.model_features import (
    feature_select_list,
    feature_source_clause,
    is_stage1_gated_sql,
    stage1_deterministic_score_sql,
)
from app.core.profile_freshness import (
    GEMINI_PROFILE_MAX_AGE_DAYS,
    max_allowed_enrichment_failures,
    needs_gemini_profile,
    needs_maps_lookup,
)
from app.services.bq_utils import execute_gemini_enrichment

logger = logging.getLogger(__name__)

PREDICTION_BQ_TIMEOUT_SECONDS = 180.0
HYBRID_TREE_WEIGHT = 0.5
HYBRID_LINEAR_WEIGHT = 0.5
LINEAR_MODEL_SUFFIX = "_linear"


def _wait_for_prediction_job(job: Any, timeout: float = PREDICTION_BQ_TIMEOUT_SECONDS) -> Any:
    try:
        return job.result(timeout=timeout)
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise


def build_prediction_input_select(project_id: str, dataset_id: str, table_ref: str,
                                  id_list_str: str) -> str:
    """The `ML.PREDICT` input: the training features, plus the join key and
    Two-Stage Hurdle Stage-1 routing columns.

    Byte-identical to `build_training_select(..., model_family='linear_reg')`'s
    feature list by construction -- both call `feature_select_list()`. The only
    additions are `m.fhrsid`, `_is_stage1_gated`, and `_stage1_capped_score`,
    which `ML.PREDICT` passes through so the MERGE can route structural
    non-candidates to their deterministic Stage-1 cap (`<= 2.0`) and plausible
    sit-down candidates to the Hybrid Ensemble prediction.
    """
    return f'''          SELECT
            m.fhrsid,
            {is_stage1_gated_sql('m', 'b')} AS _is_stage1_gated,
            {stage1_deterministic_score_sql('m')} AS _stage1_capped_score,
{feature_select_list()}
{feature_source_clause(project_id, dataset_id, table_ref)}
          WHERE m.fhrsid IN ({id_list_str})'''


def build_hybrid_merge_query(
    project_id: str,
    dataset_id: str,
    table_ref: str,
    model_ref: str,
    linear_model_ref: str,
    id_list_str: str,
) -> str:
    """Construct the Hybrid Ensemble `MERGE` query scoring against both
    `model_ref` (Course 1b Location-Blind Stage-2 Boosted Tree) and
    `linear_model_ref` (Course 2b 4:2:1 Counter-Weighted All-Scope Linear Reg)."""
    return f'''
    MERGE `{table_ref}` T
    USING (
      WITH input_features AS (
{build_prediction_input_select(project_id, dataset_id, table_ref, id_list_str)}
      ),
      tree_preds AS (
        SELECT
          fhrsid,
          _is_stage1_gated,
          _stage1_capped_score,
          predicted_user_rating AS tree_pred
        FROM ML.PREDICT(MODEL `{model_ref}`, TABLE input_features)
      ),
      lin_preds AS (
        SELECT
          fhrsid,
          predicted_user_rating AS lin_pred
        FROM ML.PREDICT(MODEL `{linear_model_ref}`, TABLE input_features)
      )
      SELECT
        t.fhrsid,
        t._is_stage1_gated,
        t._stage1_capped_score,
        t.tree_pred,
        l.lin_pred,
        ({HYBRID_TREE_WEIGHT} * t.tree_pred + {HYBRID_LINEAR_WEIGHT} * l.lin_pred) AS predicted_user_rating
      FROM tree_preds AS t
      JOIN lin_preds AS l USING (fhrsid)
    ) S
    ON T.fhrsid = S.fhrsid
    WHEN MATCHED THEN
      UPDATE SET
        predicted_user_rating = IF(
          S._is_stage1_gated,
          S._stage1_capped_score,
          ROUND(LEAST(10.0, GREATEST(1.0, S.predicted_user_rating)), 2)
        ),
        tree_pred = IF(
          S._is_stage1_gated,
          S._stage1_capped_score,
          ROUND(LEAST(10.0, GREATEST(1.0, COALESCE(S.tree_pred, S.lin_pred, 5.0))), 2)
        ),
        lin_pred = IF(
          S._is_stage1_gated,
          S._stage1_capped_score,
          ROUND(LEAST(10.0, GREATEST(1.0, COALESCE(S.lin_pred, S.tree_pred, 5.0))), 2)
        ),
        predicted_at = CURRENT_TIMESTAMP()
    '''

def build_find_query(project_id: str, dataset_id: str, table_id: str,
                     target_fhrsids: List[str] = None, limit: int = 50) -> str:
    """The batch to score, and everything needed to decide what it still costs.

    Two branches, one shape: a targeted call names its ids, an untargeted one
    takes the next unscored in-scope rows. `has_profile` is computed here rather
    than returning the profile, which is the largest column in the table and
    which nothing downstream reads.

    Extracted so `scripts/backfill_predictions.py` verifies a chunk with the
    same query that will score it a moment later, instead of a copy that can
    drift from it.

    `d_postcode` is a correlated subquery and not a `LEFT JOIN`: 15 postcodes
    are duplicated in `uk_postcode_demographics` and 103 rows of `fsa_master`
    share one, so joining returned those rows two or three times. Only "did the
    postcode resolve?" is ever read from it, and a scalar subquery answers that
    in exactly one row per restaurant -- which is also what makes `LIMIT` below
    mean what it says.
    """
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    projection = f'''
            SELECT m.fhrsid, m.postcode, m.maps_lookup_at, m.gemini_profiled_at,
                   m.gemini_insights_structured IS NOT NULL AS has_profile,
                   (SELECT MIN(d.postcode)
                    FROM `{project_id}.{dataset_id}.uk_postcode_demographics` AS d
                    WHERE REPLACE(UPPER(m.postcode), ' ', '') = REPLACE(UPPER(d.postcode), ' ', '')
                   ) AS d_postcode
            FROM `{table_ref}` AS m'''

    if target_fhrsids:
        escaped_target_ids = [fid.replace("'", "''") for fid in target_fhrsids]
        target_ids_str = ", ".join([f"'{fid}'" for fid in escaped_target_ids])
        return f'''{projection}
            WHERE m.fhrsid IN ({target_ids_str})
        '''
    return f'''{projection}
            WHERE (m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NULL AND m.predicted_user_rating IS NULL AND m.BusinessName IS NOT NULL
            LIMIT {limit}
        '''


def split_enrichment_targets(
    rows,
    force_maps: bool = False,
    force_gemini: bool = False,
    *,
    maps_max_age_days: Optional[int] = None,
    maps_cutoff_date: Optional[Any] = None,
    gemini_max_age_days: Optional[int] = GEMINI_PROFILE_MAX_AGE_DAYS,
    gemini_cutoff_date: Optional[Any] = None,
) -> dict:
    """What a batch still needs before it can be scored or trained on, by step.

    Extracted from `generate_predictions` so that `scripts/backfill_predictions.py`
    and `scripts/train_bqml_model.py` ask the exact same question.

    Takes the rows of the find/preflight query -- `fhrsid`, `maps_lookup_at`,
    `gemini_profiled_at`, `has_profile`, `postcode`, `d_postcode` -- and returns
    the three lists plus `never_profiled` and `never_looked_up_maps`.
    """
    fhrsids = [str(row.fhrsid) for row in rows]

    # `needs_maps_lookup` reads `maps_lookup_at`, not `maps_rating`: Phase 5
    # retired the `-1` sentinel, so a NULL rating no longer distinguishes
    # "never looked up" from "looked up, Places had nothing". Once a staleness
    # threshold (`maps_max_age_days` or `maps_cutoff_date`) is set, old lookups
    # (including previous misses) are retried.
    maps_missing = [
        str(row.fhrsid) for row in rows
        if needs_maps_lookup(
            getattr(row, 'maps_lookup_at', None),
            force=force_maps,
            max_age_days=maps_max_age_days,
            cutoff_date=maps_cutoff_date,
        )
    ]

    # One predicate, shared with the UI's cost estimate and the training pre-flight.
    gemini_missing = [
        str(row.fhrsid) for row in rows
        if needs_gemini_profile(
            row.has_profile,
            getattr(row, 'gemini_profiled_at', None),
            force=force_gemini,
            max_age_days=gemini_max_age_days,
            cutoff_date=gemini_cutoff_date,
        )
    ]

    # A row with no postcode at all cannot be looked up, so it is not queued;
    # only one whose postcode failed to join the demographics table is.
    postcodes_missing = [
        str(row.fhrsid) for row in rows
        if getattr(row, 'd_postcode', None) is None and getattr(row, 'postcode', None) is not None
    ]

    return {
        'fhrsids': fhrsids,
        'maps': maps_missing,
        'gemini': gemini_missing,
        'postcodes': postcodes_missing,
        'never_profiled': sum(1 for row in rows if not row.has_profile),
        'never_looked_up_maps': sum(1 for row in rows if getattr(row, 'maps_lookup_at', None) is None),
    }


def generate_predictions(
    project_id: str,
    dataset_id: str,
    table_id: str,
    model_name: str,
    limit: int = 50,
    target_fhrsids: List[str] = None,
    force_maps: bool = False,
    force_gemini: bool = False,
    *,
    maps_max_age_days: Optional[int] = None,
    maps_cutoff_date: Optional[Any] = None,
    gemini_max_age_days: Optional[int] = GEMINI_PROFILE_MAX_AGE_DAYS,
    gemini_cutoff_date: Optional[Any] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    model_ref = f"{project_id}.{dataset_id}.{model_name}"

    if progress_callback:
        progress_callback("🔎 Auditing freshness of target prediction batch...")

    # Step 1: Identify target batch
    find_query = build_find_query(project_id, dataset_id, table_id,
                                  target_fhrsids=target_fhrsids, limit=limit)
    try:
        results = _wait_for_prediction_job(client.query(find_query))
        rows = list(results)
        split = split_enrichment_targets(
            rows,
            force_maps=force_maps,
            force_gemini=force_gemini,
            maps_max_age_days=maps_max_age_days,
            maps_cutoff_date=maps_cutoff_date,
            gemini_max_age_days=gemini_max_age_days,
            gemini_cutoff_date=gemini_cutoff_date,
        )
        fhrsids = split['fhrsids']
        maps_missing_fhrsids = split['maps']
        gemini_missing_fhrsids = split['gemini']
        never_profiled = split['never_profiled']
        never_maps = split['never_looked_up_maps']
        postcodes_missing = split['postcodes']
    except Exception as e:
        logger.error(f"Error finding target batch: {e}")
        return False, f"Failed to identify target batch: {str(e)}"

    if not fhrsids:
        return True, "No pending restaurants require predictions."

    if progress_callback:
        gemini_batches = (len(gemini_missing_fhrsids) + 24) // 25
        progress_callback(
            f"📋 Audit complete ({len(fhrsids)} target restaurant(s)): "
            f"{len(maps_missing_fhrsids)} Maps lookups · "
            f"{len(gemini_missing_fhrsids)} Gemini profiles ({gemini_batches} batch(es) of 25) · "
            f"{len(postcodes_missing)} Postcode lookups."
        )

    # Step 2a: Auto-enrichment Maps (must complete before Gemini and ML.PREDICT)
    if maps_missing_fhrsids:
        refreshed_maps = len(maps_missing_fhrsids) - min(never_maps, len(maps_missing_fhrsids))
        logger.info(
            f"Running maps enrichment for {len(maps_missing_fhrsids)} restaurants "
            f"({never_maps} never looked up, {refreshed_maps} stale or forced)."
        )
        if progress_callback:
            progress_callback(
                f"🗺️ Regenerating Google Maps data for {len(maps_missing_fhrsids)} restaurant(s) "
                f"({never_maps} missing, {refreshed_maps} stale/forced)..."
            )
        maps_force_regen = bool(
            force_maps or maps_max_age_days is not None or maps_cutoff_date is not None
        )
        try:
            maps_kwargs: dict[str, Any] = {
                "limit": len(maps_missing_fhrsids),
                "force_regen": maps_force_regen,
            }
            if progress_callback is not None:
                maps_kwargs["progress_callback"] = progress_callback
            updated_maps = enrich_restaurants_by_fhrsid(
                maps_missing_fhrsids,
                **maps_kwargs,
            )
        except Exception as e:
            logger.error(f"Maps Auto-enrichment failed: {e}")
            return False, f"Pre-prediction Maps enrichment failed: {e}"

        if isinstance(updated_maps, int):
            allowed_maps = max_allowed_enrichment_failures(len(maps_missing_fhrsids))
            shortfall = len(maps_missing_fhrsids) - updated_maps
            if shortfall > allowed_maps:
                msg = (
                    f"Pre-prediction Maps enrichment updated only {updated_maps}/{len(maps_missing_fhrsids)} "
                    f"restaurants ({shortfall} unrefreshed, allowed <= {allowed_maps})."
                )
                logger.error(msg)
                return False, msg
    elif progress_callback:
        progress_callback(
            f"🗺️ Google Maps data: all {len(fhrsids)} target restaurant(s) are already fresh (0 to regenerate)."
        )

    # Step 2b: Auto-enrichment Gemini Insights
    if gemini_missing_fhrsids:
        # Split the count: a refresh of an existing profile is a cost decision,
        # and it should be visible in the log that one happened.
        refreshed = len(gemini_missing_fhrsids) - min(never_profiled, len(gemini_missing_fhrsids))
        gemini_batches = (len(gemini_missing_fhrsids) + 24) // 25
        logger.info(
            f"Running Gemini enrichment for {len(gemini_missing_fhrsids)} restaurants "
            f"({never_profiled} never profiled, {refreshed} stale or forced).")
        if progress_callback:
            progress_callback(
                f"✨ Regenerating Gemini profiles for {len(gemini_missing_fhrsids)} restaurant(s) "
                f"across {gemini_batches} batch(es) ({never_profiled} missing, {refreshed} stale/forced)..."
            )
        try:
            gemini_kwargs: dict[str, Any] = {"fhrsids": gemini_missing_fhrsids}
            if progress_callback is not None:
                gemini_kwargs["progress_callback"] = progress_callback
            gemini_ok = execute_gemini_enrichment(
                project_id, dataset_id, table_id, **gemini_kwargs
            )
        except Exception as e:
            logger.error(f"Gemini Auto-enrichment failed: {e}")
            return False, f"Pre-prediction Gemini enrichment failed: {e}"
        if gemini_ok is False:
            msg = (
                f"Pre-prediction Gemini enrichment failed or timed out for "
                f"{len(gemini_missing_fhrsids)} restaurant(s)."
            )
            logger.error(msg)
            return False, msg
    elif progress_callback:
        progress_callback(
            f"✨ Gemini profiles: all {len(fhrsids)} target restaurant(s) are already fresh (0 to regenerate)."
        )

    # Step 2c: Auto-enrichment Postcode Demographics
    if postcodes_missing:
        logger.info(f"Running Postcode Demographics enrichment for {len(postcodes_missing)} restaurants.")
        if progress_callback:
            progress_callback(
                f"📮 Enriching UK postcode demographics for {len(postcodes_missing)} restaurant(s)..."
            )
        try:
            from scripts.enrich_postcode_demographics import enrich_postcodes
            enrich_postcodes(project_id=project_id, dataset_id=dataset_id, master_table=table_id)
        except Exception as e:
            logger.error(f"Postcode Demographics Auto-enrichment failed: {e}")
            return False, f"Pre-prediction Postcode enrichment failed: {e}"

    # Step 3: Run Prediction
    if progress_callback:
        progress_callback(f"⚡ Scoring {len(fhrsids)} restaurant(s) via BigQuery ML.PREDICT...")

    escaped_ids = [fid.replace("'", "''") for fid in fhrsids]
    id_list_str = ", ".join([f"'{fid}'" for fid in escaped_ids])
    linear_model_name = (
        model_name
        if model_name.endswith(LINEAR_MODEL_SUFFIX)
        else f"{model_name}{LINEAR_MODEL_SUFFIX}"
    )
    linear_model_ref = f"{project_id}.{dataset_id}.{linear_model_name}"

    predict_query = build_hybrid_merge_query(
        project_id,
        dataset_id,
        table_ref,
        model_ref,
        linear_model_ref,
        id_list_str,
    )

    try:
        job = client.query(predict_query)
        _wait_for_prediction_job(job)
        updated_rows = job.num_dml_affected_rows
        if progress_callback:
            progress_callback(f"✅ Scored and updated {updated_rows} restaurant(s) in BigQuery.")
        return True, f"Successfully predicted ratings for {updated_rows} restaurants."
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False, f"Prediction failed: {str(e)}"
