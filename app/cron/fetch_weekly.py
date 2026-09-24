import logging
import os
from google.cloud import bigquery
import pandas as pd
from typing import List, Tuple

from app.core.data_processing import (
    parse_coordinates,
    fetch_data_for_all_coordinates,
    process_and_update_master_data,
    parse_bq_path
)
from app.services.bq_utils import (
    BigQueryExecutionError,
    load_fhrsids_from_bq,
    append_to_bigquery,
    MASTER_BQ_SCHEMA,
    ORIGINAL_COLUMNS_TO_KEEP
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config details
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants"
CONFIG_TABLE_ID = "config_search_params"

def get_config_params() -> List[dict]:
    """Reads configuration parameters from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{CONFIG_TABLE_ID}"
    query = f"SELECT * FROM `{table_ref}`"
    
    logger.info(f"Reading config from {table_ref}")
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        config_list = []
        for row in results:
            config_list.append(dict(row))
        return config_list
    except Exception as e:
        logger.error(f"Error reading config table: {e}")
        return []

def run_sync_for_config(config: dict):
    """Runs synchronization for a single configuration row."""
    lat = config.get('latitude')
    lon = config.get('longitude')
    max_results = config.get('max_results')
    target_bq_path = config.get('target_bq_table')
    
    logger.info(f"Starting sync for target: {target_bq_path} with max_results: {max_results}")
    
    try:
        project_id, dataset_id, table_id = parse_bq_path(target_bq_path)
    except ValueError as e:
        logger.error(f"Invalid target BQ path: {target_bq_path}. Error: {e}")
        return

    # 1. Parse/Validate Coordinates
    if lat is None or lon is None:
        logger.error(f"Missing latitude or longitude in config: {config}")
        return

    try:
        # fetch_data_for_all_coordinates expects (lon, lat)
        valid_coords = [(float(lon), float(lat))]
    except ValueError as e:
        logger.error(f"Invalid coordinate values: lat={lat}, lon={lon}. Error: {e}")
        return

    # 2. Fetch API Data
    logger.info("Fetching data from API...")
    all_api_establishments = fetch_data_for_all_coordinates(valid_coords, max_results)
    logger.info(f"Fetched {len(all_api_establishments)} records from API.")
    
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}

    # 3. Load the IDs we already hold -- nothing else is needed to spot new records
    logger.info("Loading existing FHRSIDs from BigQuery...")
    # Deliberately not caught. Aborting is right -- appending without knowing
    # what already exists would duplicate the table -- but swallowing it left
    # `main()` reporting a successful week in which nothing was ingested (D9).
    existing_fhrsids = load_fhrsids_from_bq(project_id, dataset_id, table_id)
    logger.info(f"Loaded {len(existing_fhrsids)} existing records.")

    # 4. Process and Identify New
    logger.info("Processing data to identify new records...")
    new_restaurants, summary_msg = process_and_update_master_data(existing_fhrsids, combined_api_data)
    logger.info(f"Process summary: {summary_msg}")

    if not new_restaurants:
        logger.info("No new restaurants to append.")
        return

    # 5. Append to BigQuery
    logger.info(f"Appending {len(new_restaurants)} new records to BigQuery...")
    df_new = pd.DataFrame(new_restaurants)
    
    # Normalize columns to lowercase to match BQ schema
    df_new.columns = [c.lower() for c in df_new.columns]
    
    # Ensure columns match schema
    success = append_to_bigquery(
        df=df_new,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        bq_schema=MASTER_BQ_SCHEMA
    )
    
    if not success:
        # `append_to_bigquery` reports failure by returning False. Logging it
        # and returning meant the new restaurants were dropped on the floor and
        # the run still counted as a success.
        raise BigQueryExecutionError(
            f"Failed to append {len(new_restaurants)} new records to {table_id}")
    logger.info("Append successful.")

def main():
    """Run every configured search area, and tell the truth about the result.

    Exit status is the only thing Cloud Run reads. This used to catch
    everything and return, so the job exited 0 whatever happened -- a weekly
    ingest could fail on an expired credential for a month, and the only
    evidence would be log lines nobody opens and a `first_seen` gap nobody
    would attribute to this (D9).

    One bad search area still does not cost the others their run; the failures
    are collected and reported together at the end.
    """
    logger.info("Starting Weekly Fetch Job")
    try:
        configs = get_config_params()
    except Exception as e:
        logger.exception(f"Could not read the search configuration: {e}")
        raise SystemExit(f"Weekly fetch aborted: could not read the search configuration: {e}")

    if not configs:
        # Not a failure. Emptying the config table is how the cron gets paused,
        # and a paused job that fails every week would just get muted.
        logger.warning("No configuration found in config table.")
        return

    failures = []
    for config in configs:
        try:
            run_sync_for_config(config)
        except Exception as e:
            logger.exception(f"Error processing config {config}: {e}")
            failures.append(e)

    if failures:
        raise SystemExit(
            f"Weekly fetch: {len(failures)} of {len(configs)} search areas failed. "
            f"First error: {failures[0]}")

    logger.info(f"Weekly Fetch Job complete: {len(configs)} search areas processed.")

if __name__ == "__main__":
    main()
