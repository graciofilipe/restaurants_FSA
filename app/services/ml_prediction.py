import logging
from google.cloud import bigquery
from typing import Tuple
from scripts.enrich_maps_data import enrich_restaurants_by_fhrsid
from app.services.bq_utils import execute_gemini_enrichment

logger = logging.getLogger(__name__)

def generate_predictions(project_id: str, dataset_id: str, table_id: str, model_name: str, limit: int = 50) -> Tuple[bool, str]:
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    model_ref = f"{project_id}.{dataset_id}.{model_name}"

    # Step 1: Identify target batch
    find_query = f'''
        SELECT fhrsid, maps_rating, gemini_insights
        FROM `{table_ref}`
        WHERE user_rating IS NULL AND predicted_user_rating IS NULL AND BusinessName IS NOT NULL
        LIMIT {limit}
    '''
    try:
        results = client.query(find_query).result()
        rows = list(results)
        fhrsids = [str(row.fhrsid) for row in rows]
        maps_missing_fhrsids = [str(row.fhrsid) for row in rows if row.maps_rating is None]
        gemini_missing_fhrsids = [str(row.fhrsid) for row in rows if row.gemini_insights is None]
    except Exception as e:
        logger.error(f"Error finding target batch: {e}")
        return False, f"Failed to identify target batch: {str(e)}"

    if not fhrsids:
        return True, "No pending restaurants require predictions."

    # Step 2a: Auto-enrichment Maps
    if maps_missing_fhrsids:
        logger.info(f"Running maps enrichment for {len(maps_missing_fhrsids)} restaurants.")
        try:
            enrich_restaurants_by_fhrsid(maps_missing_fhrsids, limit=len(maps_missing_fhrsids))
        except Exception as e:
            logger.warning(f"Maps Auto-enrichment encountered an error: {e}")

    # Step 2b: Auto-enrichment Gemini Insights
    if gemini_missing_fhrsids:
        logger.info(f"Running Gemini enrichment for {len(gemini_missing_fhrsids)} restaurants.")
        try:
            execute_gemini_enrichment(project_id, dataset_id, table_id, fhrsids=gemini_missing_fhrsids)
        except Exception as e:
            logger.warning(f"Gemini Auto-enrichment encountered an error: {e}")

    # Step 3: Run Prediction
    escaped_ids = [fid.replace("'", "''") for fid in fhrsids]
    id_list_str = ", ".join([f"'{fid}'" for fid in escaped_ids])

    predict_query = f'''
    MERGE `{table_ref}` T
    USING (
      SELECT fhrsid, predicted_user_rating FROM ML.PREDICT(MODEL `{model_ref}`,
        (
          SELECT
            fhrsid,
            postcode,
            localauthorityname,
            ratingvalue,
            user_rating,
            price_level,
            maps_rating,
            maps_reviews,
            latitude,
            longitude,
            business_status,
            SPLIT(REPLACE(maps_types, ' ', ''), ',') AS maps_types_array,
            IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.1_value_and_volume_rating') AS INT64), 0) AS score_1_value_and_volume_rating,
            IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.2_demographic_community_score') AS INT64), 0) AS score_2_demographic_community_score,
            IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.3_linguistic_signal_score') AS INT64), 0) AS score_3_linguistic_signal_score,
            IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.4_geographic_precision_specificity_level') AS INT64), 0) AS score_4_geographic_precision_specificity_level,
            IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.5_culinary_uncompromisingness_score') AS INT64), 0) AS score_5_culinary_uncompromisingness_score,
            IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.match_score') AS INT64), 0) AS match_score
          FROM `{table_ref}`
          WHERE fhrsid IN ({id_list_str})
        )
      )
    ) S
    ON T.fhrsid = S.fhrsid
    WHEN MATCHED THEN
      UPDATE SET predicted_user_rating = S.predicted_user_rating
    '''

    try:
        job = client.query(predict_query)
        job.result()
        updated_rows = job.num_dml_affected_rows
        return True, f"Successfully predicted ratings for {updated_rows} restaurants."
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False, f"Prediction failed: {str(e)}"
