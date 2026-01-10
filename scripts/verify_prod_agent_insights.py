from google.cloud import bigquery
import os
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants"
TABLE_ID = "restaurant_agent_insights"

def verify_production_insights():
    """
    Queries the restaurant_agent_insights table for the most recent entry.
    """
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    print(f"Checking table: {table_ref} for recent updates...")
    
    query = f"""
        SELECT *
        FROM `{table_ref}`
        ORDER BY updated_at DESC
        LIMIT 5
    """
    
    try:
        query_job = client.query(query)
        results = list(query_job.result())
        
        if not results:
            print("No insights found in the table.")
            return

        print(f"\nFound {len(results)} recent insights:")
        print("-" * 50)
        
        current_time = datetime.now(timezone.utc)
        
        for row in results:
            updated_at = row['updated_at']
            time_diff = current_time - updated_at
            
            print(f"FHRSID: {row['fhrsid']}")
            print(f"Cuisine: {row['cuisine_type']}")
            print(f"Reviews: {row['review_count']}")
            print(f"Rating: {row['average_rating']}")
            print(f"Updated At: {updated_at} (Age: {time_diff})")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error querying BigQuery: {e}")

if __name__ == "__main__":
    verify_production_insights()
