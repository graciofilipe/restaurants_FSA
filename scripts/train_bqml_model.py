import argparse
import logging
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(
    project_id: str, 
    dataset_id: str, 
    table_id: str, 
    model_name: str,
    dry_run: bool = False,
    run_async: bool = False
):
    """
    Constructs and executes a BQML model training query.
    """
    client = bigquery.Client(project=project_id)
    
    full_model_name = f"{project_id}.{dataset_id}.{model_name}"
    source_table = f"{project_id}.{dataset_id}.{table_id}"
    
    # Pre-flight JIT Enrichment: check all labeled examples for missing features.
    logger.info("Executing pre-flight JIT check for labeled training examples...")
    check_query = f"""
        SELECT m.fhrsid, m.postcode, m.maps_rating, m.gemini_insights_structured, d.postcode AS d_postcode
        FROM `{source_table}` AS m
        LEFT JOIN `{project_id}.{dataset_id}.uk_postcode_demographics` AS d
          ON REPLACE(UPPER(m.postcode), ' ', '') = REPLACE(UPPER(d.postcode), ' ', '')
        WHERE (m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NOT NULL
    """
    try:
        results = client.query(check_query).result()
        rows = list(results)
        
        maps_missing = [str(row.fhrsid) for row in rows if row.maps_rating is None]
        gemini_missing = [str(row.fhrsid) for row in rows if row.gemini_insights_structured is None]
        postcode_missing = [str(row.fhrsid) for row in rows if getattr(row, 'd_postcode', None) is None and getattr(row, 'postcode', None) is not None]
        
        if maps_missing:
            logger.info(f"JIT: Found {len(maps_missing)} labeled restaurants missing Maps data. Triggering enrichment...")
            from scripts.enrich_maps_data import enrich_restaurants_by_fhrsid
            enrich_restaurants_by_fhrsid(maps_missing, limit=len(maps_missing))
            
        if gemini_missing:
            logger.info(f"JIT: Found {len(gemini_missing)} labeled restaurants missing Gemini insights. Triggering enrichment...")
            from app.services.bq_utils import execute_gemini_enrichment
            execute_gemini_enrichment(project_id, dataset_id, table_id, fhrsids=gemini_missing)
            
        if postcode_missing:
            logger.info("JIT: Found labeled restaurants missing postcode demographics. Triggering enrichment...")
            from scripts.enrich_postcode_demographics import enrich_postcodes
            enriched_postcodes = enrich_postcodes(project_id=project_id, dataset_id=dataset_id, master_table=table_id)
            if enriched_postcodes > 0:
                logger.info(f"JIT: Enriched {enriched_postcodes} new postcodes with demographic data.")
            
    except Exception as e:
        logger.warning(f"JIT pre-flight check failed: {e}. Proceeding with existing data.")
        
    # We omit BusinessType as it is not present in the BigQuery schema for fsa_master.
    query = f"""
    CREATE OR REPLACE MODEL `{full_model_name}`
    OPTIONS(
      model_type='BOOSTED_TREE_REGRESSOR',
      input_label_cols=['user_rating'],
      model_registry='vertex_ai'
    ) AS
    SELECT
      m.postcode,
      m.localauthorityname,
      m.ratingvalue,
      m.user_rating,
      m.price_level,
      m.maps_rating,
      m.maps_reviews,
      m.latitude,
      m.longitude,
      m.business_status,
      SPLIT(REPLACE(m.maps_types, ' ', ''), ',') AS maps_types_array,
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.1_value_and_volume_rating') AS INT64), 0) AS score_1_value_and_volume_rating,
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.2_demographic_community_score') AS INT64), 0) AS score_2_demographic_community_score,
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.3_linguistic_signal_score') AS INT64), 0) AS score_3_linguistic_signal_score,
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.4_geographic_precision_specificity_level') AS INT64), 0) AS score_4_geographic_precision_specificity_level,
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.5_culinary_uncompromisingness_score') AS INT64), 0) AS score_5_culinary_uncompromisingness_score,
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.match_score') AS INT64), 0) AS match_score,
      d.lsoa,
      d.msoa,
      d.imd_rank
    FROM
      `{source_table}` AS m
    LEFT JOIN
      `{project_id}.{dataset_id}.uk_postcode_demographics` AS d
    ON
      REPLACE(UPPER(m.postcode), ' ', '') = REPLACE(UPPER(d.postcode), ' ', '')
    WHERE
      (m.in_scope = TRUE OR m.in_scope IS NULL)
      AND m.user_rating IS NOT NULL
    """
    
    logger.info(f"Preparing BQML Training Query for {full_model_name}...")
    if dry_run:
        logger.info("Executing DRY RUN to validate query without training.")
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            query_job = client.query(query, job_config=job_config)
            logger.info("Dry run successful. Query is valid.")
            logger.info(f"This query will process {query_job.total_bytes_processed} bytes.")
        except GoogleCloudError as e:
            logger.error(f"BigQuery validation failed: {e}")
            raise
    else:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BQML Restaurant Preference Model")
    parser.add_argument("--project_id", default="filipegracio-ai-learning", help="GCP Project ID")
    parser.add_argument("--dataset_id", default="filipegracio_fsa_restaurants", help="BigQuery Dataset ID")
    parser.add_argument("--table_id", default="fsa_master", help="Source Table ID")
    parser.add_argument("--model_name", default="restaurant_preference_model", help="Target Model Name")
    parser.add_argument("--dry-run", action="store_true", help="Validate query without executing training")
    parser.add_argument("--run_async", action="store_true", help="Run model training asynchronously")
    args = parser.parse_args()
    
    train_model(args.project_id, args.dataset_id, args.table_id, args.model_name, dry_run=args.dry_run, run_async=args.run_async)
