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
    dry_run: bool = False
):
    """
    Constructs and executes a BQML model training query.
    """
    client = bigquery.Client(project=project_id)
    
    full_model_name = f"{project_id}.{dataset_id}.{model_name}"
    source_table = f"{project_id}.{dataset_id}.{table_id}"
    
    # We omit BusinessType as it is not present in the BigQuery schema for fsa_master.
    query = f"""
    CREATE OR REPLACE MODEL `{full_model_name}`
    OPTIONS(
      model_type='BOOSTED_TREE_REGRESSOR',
      input_label_cols=['user_rating']
    ) AS
    WITH parsed_data AS (
      SELECT
        *,
        REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]') AS clean_json
      FROM
        `{source_table}`
    )
    SELECT
      postcode,
      localauthorityname,
      ratingvalue,
      user_rating,
      CASE
        WHEN gemini_insights IS NOT NULL OR gemini_insights_structured IS NOT NULL THEN 1
        ELSE 0
      END AS has_gemini_insights,
      CAST(JSON_VALUE(clean_json, '$."match_score"') AS FLOAT64) AS `match_score`,
      CAST(JSON_VALUE(clean_json, '$."1_value_and_volume".rating') AS FLOAT64) AS `1_value_and_volume_rating`,
      CAST(JSON_VALUE(clean_json, '$."2_demographic_community".score') AS FLOAT64) AS `2_demographic_community_score`,
      CAST(JSON_VALUE(clean_json, '$."3_linguistic_signal".score') AS FLOAT64) AS `3_linguistic_signal_score`,
      JSON_VALUE(clean_json, '$."4_geographic_precision".region_identified') AS `4_geographic_precision_region_identified`,
      JSON_VALUE(clean_json, '$."4_geographic_precision".specificity_level') AS `4_geographic_precision_specificity_level`,
      CAST(JSON_VALUE(clean_json, '$."5_culinary_uncompromisingness".score') AS FLOAT64) AS `5_culinary_uncompromisingness_score`,
      CAST(JSON_VALUE(clean_json, '$."6_establishment_integrity".is_sit_down_restaurant') AS BOOL) AS `6_establishment_integrity_is_sit_down_restaurant`,
      JSON_VALUE(clean_json, '$."6_establishment_integrity".type') AS `6_establishment_integrity_type`
    FROM
      parsed_data
    WHERE
      user_rating IS NOT NULL
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
            query_job.result()  # Wait for the job to complete
            logger.info(f"Model {full_model_name} trained successfully.")
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
    args = parser.parse_args()
    
    train_model(args.project_id, args.dataset_id, args.table_id, args.model_name, dry_run=args.dry_run)
