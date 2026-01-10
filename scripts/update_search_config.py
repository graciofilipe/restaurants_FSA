from google.cloud import bigquery
import pandas as pd
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_search_config(project_id: str, dataset_id: str, table_id: str, coordinates: List[Tuple[float, float]], max_results: int = 5000) -> bool:
    """
    Adds new search coordinates to the configuration table.
    
    Args:
        project_id: GCP Project ID
        dataset_id: BigQuery Dataset ID
        table_id: BigQuery Table ID
        coordinates: List of (longitude, latitude) tuples
        max_results: Max results to fetch for these points (default 5000)
        
    Returns:
        bool: True if successful, False otherwise.
    """
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    rows = []
    target_bq_table = f"{project_id}.{dataset_id}.fsa_master"
    
    for lon, lat in coordinates:
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "radius": None, # Using None/NA as per existing schema inspection
            "max_results": max_results,
            "target_bq_table": target_bq_table
        })
        
    if not rows:
        logger.warning("No coordinates provided.")
        return False
        
    df = pd.DataFrame(rows)
    
    try:
        # Get table to ensure it exists and fetch schema
        table = client.get_table(table_ref)
        
        # Insert rows
        # Using insert_rows_from_dataframe is cleaner for pandas
        # We must pass the schema to avoid the "Could not determine schema" error if the DataFrame types are ambiguous
        errors = client.insert_rows_from_dataframe(table, df)
        
        if not errors:
            logger.info(f"Successfully added {len(rows)} new search configurations.")
            return True
        else:
            # Flatten error list if needed or print the first few
            logger.error(f"Encountered errors while inserting rows: {errors}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update config table: {e}")
        return False

if __name__ == "__main__":
    # Placeholder for CLI usage
    pass
