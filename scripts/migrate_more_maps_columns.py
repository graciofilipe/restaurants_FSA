import os
from google.cloud import bigquery

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
BQ_PATH = os.environ.get("BQ_PATH", DEFAULT_BQ_PATH)

def main():
    project_id, dataset_id, table_id = BQ_PATH.split(".")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    print(f"Adding new maps columns to {table_ref}...")
    
    alter_query = f"""
    ALTER TABLE `{table_ref}`
    ADD COLUMN IF NOT EXISTS latitude FLOAT64,
    ADD COLUMN IF NOT EXISTS longitude FLOAT64,
    ADD COLUMN IF NOT EXISTS maps_url STRING,
    ADD COLUMN IF NOT EXISTS business_status STRING,
    ADD COLUMN IF NOT EXISTS website_url STRING,
    ADD COLUMN IF NOT EXISTS maps_types STRING
    """
    
    try:
        job = client.query(alter_query)
        job.result()
        print("Successfully added columns (or they already existed).")
    except Exception as e:
        print(f"Error adding columns: {e}")

if __name__ == "__main__":
    main()
