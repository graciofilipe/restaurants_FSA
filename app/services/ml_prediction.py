import logging
from google.cloud import bigquery
from typing import Tuple, List
from scripts.enrich_maps_data import enrich_restaurants_by_fhrsid
from app.core.model_features import feature_select_list, feature_source_clause
from app.core.profile_freshness import needs_gemini_profile
from app.services.bq_utils import execute_gemini_enrichment

logger = logging.getLogger(__name__)


def build_prediction_input_select(project_id: str, dataset_id: str, table_ref: str,
                                  id_list_str: str) -> str:
    """The `ML.PREDICT` input: the training features, plus the join key.

    Byte-identical to `build_training_select`'s feature list by construction --
    both call `feature_select_list()`. The only addition is `m.fhrsid`, which
    ML.PREDICT passes through so the MERGE has something to match on; it is not
    a feature and the model never saw it.

    Train/serve skew here is silent: BigQuery would happily predict from a
    differently-computed column and write a plausible-looking number.
    """
    return f'''          SELECT
            m.fhrsid,
{feature_select_list()}
{feature_source_clause(project_id, dataset_id, table_ref)}
          WHERE m.fhrsid IN ({id_list_str})'''

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


def split_enrichment_targets(rows, force_maps: bool = False,
                             force_gemini: bool = False) -> dict:
    """What a batch still needs before it can be scored, by enrichment step.

    Extracted from `generate_predictions` so that `scripts/backfill_predictions.py`
    can ask the same question in order to *refuse* to run when the answer is not
    "nothing". A guard that re-derives the predicate is a guard that can disagree
    with the thing it guards, which is D1 with the cost moved elsewhere.

    Takes the rows of the find query -- `fhrsid`, `maps_lookup_at`,
    `gemini_profiled_at`, `has_profile`, `postcode`, `d_postcode` -- and returns
    the three lists plus `never_profiled`, which splits the Gemini count into
    first profiles and refreshes because those are different cost decisions.
    """
    fhrsids = [str(row.fhrsid) for row in rows]

    if force_maps:
        maps_missing = fhrsids.copy()
    else:
        # `maps_lookup_at`, not `maps_rating`: Phase 5 retired the `-1`
        # sentinel, so a NULL rating no longer distinguishes "never looked
        # up" from "looked up, Places had nothing". Testing the rating
        # would re-query 243 permanent misses on every run, at cost.
        maps_missing = [str(row.fhrsid) for row in rows if row.maps_lookup_at is None]

    # One predicate, shared with the UI's cost estimate: an estimate
    # computed differently from the spend it predicts is D1. The query
    # returns `has_profile` rather than the profile itself -- nothing here
    # reads the JSON, and it is the largest column in the table.
    gemini_missing = [
        str(row.fhrsid) for row in rows
        if needs_gemini_profile(row.has_profile, row.gemini_profiled_at, force=force_gemini)
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
    }


def generate_predictions(project_id: str, dataset_id: str, table_id: str, model_name: str, limit: int = 50, target_fhrsids: List[str] = None, force_maps: bool = False, force_gemini: bool = False) -> Tuple[bool, str]:
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    model_ref = f"{project_id}.{dataset_id}.{model_name}"

    # Step 1: Identify target batch
    find_query = build_find_query(project_id, dataset_id, table_id,
                                  target_fhrsids=target_fhrsids, limit=limit)
    try:
        results = client.query(find_query).result()
        rows = list(results)
        split = split_enrichment_targets(rows, force_maps=force_maps, force_gemini=force_gemini)
        fhrsids = split['fhrsids']
        maps_missing_fhrsids = split['maps']
        gemini_missing_fhrsids = split['gemini']
        never_profiled = split['never_profiled']
        postcodes_missing = split['postcodes']
    except Exception as e:
        logger.error(f"Error finding target batch: {e}")
        return False, f"Failed to identify target batch: {str(e)}"

    if not fhrsids:
        return True, "No pending restaurants require predictions."

    # Step 2a: Auto-enrichment Maps
    if maps_missing_fhrsids:
        logger.info(f"Running maps enrichment for {len(maps_missing_fhrsids)} restaurants.")
        try:
            enrich_restaurants_by_fhrsid(maps_missing_fhrsids, limit=len(maps_missing_fhrsids), force_regen=force_maps)
        except Exception as e:
            logger.warning(f"Maps Auto-enrichment encountered an error: {e}")

    # Step 2b: Auto-enrichment Gemini Insights
    if gemini_missing_fhrsids:
        # Split the count: a refresh of an existing profile is a cost decision,
        # and it should be visible in the log that one happened.
        refreshed = len(gemini_missing_fhrsids) - min(never_profiled, len(gemini_missing_fhrsids))
        logger.info(
            f"Running Gemini enrichment for {len(gemini_missing_fhrsids)} restaurants "
            f"({never_profiled} never profiled, {refreshed} stale or forced).")
        try:
            execute_gemini_enrichment(project_id, dataset_id, table_id, fhrsids=gemini_missing_fhrsids)
        except Exception as e:
            logger.warning(f"Gemini Auto-enrichment encountered an error: {e}")

    # Step 2c: Auto-enrichment Postcode Demographics
    if postcodes_missing:
        logger.info(f"Running Postcode Demographics enrichment for {len(postcodes_missing)} restaurants.")
        try:
            from scripts.enrich_postcode_demographics import enrich_postcodes
            enrich_postcodes(project_id=project_id, dataset_id=dataset_id, master_table=table_id)
        except Exception as e:
            logger.warning(f"Postcode Demographics Auto-enrichment encountered an error: {e}")

    # Step 3: Run Prediction
    escaped_ids = [fid.replace("'", "''") for fid in fhrsids]
    id_list_str = ", ".join([f"'{fid}'" for fid in escaped_ids])

    predict_query = f'''
    MERGE `{table_ref}` T
    USING (
      SELECT fhrsid, predicted_user_rating FROM ML.PREDICT(MODEL `{model_ref}`,
        (
{build_prediction_input_select(project_id, dataset_id, table_ref, id_list_str)}
        )
      )
    ) S
    ON T.fhrsid = S.fhrsid
    WHEN MATCHED THEN
      UPDATE SET
        predicted_user_rating = S.predicted_user_rating,
        predicted_at = CURRENT_TIMESTAMP()
    '''

    try:
        job = client.query(predict_query)
        job.result()
        updated_rows = job.num_dml_affected_rows
        return True, f"Successfully predicted ratings for {updated_rows} restaurants."
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False, f"Prediction failed: {str(e)}"
