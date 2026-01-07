import logging
import os
from google.cloud import bigquery
import pandas as pd
from typing import List, Tuple

from app.core.data_processing import (
    parse_coordinates,
    fetch_data_for_all_coordinates,
    load_master_data,
    process_and_update_master_data,
    parse_bq_path
)
from app.services.bq_utils import (
    load_all_data_from_bq,
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

    # 3. Load Master Data
    logger.info("Loading master data from BigQuery...")
    try:
        # load_master_data expects a loader function
        master_restaurant_data = load_master_data(project_id, dataset_id, table_id, load_all_data_from_bq)
        logger.info(f"Loaded {len(master_restaurant_data)} existing records.")
    except Exception as e:
        logger.error(f"Failed to load master data: {e}")
        return

    # 4. Process and Identify New
    logger.info("Processing data to identify new records...")
    new_restaurants, summary_msg = process_and_update_master_data(master_restaurant_data, combined_api_data)
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
    
    if success:
        logger.info("Append successful.")
    else:
        logger.error("Append failed.")

def main():
    logger.info("Starting Weekly Fetch Job")
    try:
        configs = get_config_params()
        if not configs:
            logger.warning("No configuration found in config table.")
            return

        for config in configs:
            try:
                run_sync_for_config(config)
            except Exception as e:
                logger.error(f"Error processing config {config}: {e}")
                continue
            
    except Exception as e:
        logger.exception(f"Fatal error in fetch_weekly job: {e}")

if __name__ == "__main__":
    main()
