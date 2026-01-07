from google.cloud import bigquery
import os

# Default Project ID - ideally this should be fetched from environment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants" 
CONFIG_TABLE_ID = "config_search_params"

# Updated Schema definition
schema = [
    bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED", description="Search center latitude"),
    bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED", description="Search center longitude"),
    bigquery.SchemaField("radius", "INTEGER", mode="NULLABLE", description="Search radius in miles (default applied if null)"),
    bigquery.SchemaField("max_results", "INTEGER", mode="REQUIRED", description="API fetch limit per coordinate"),
    bigquery.SchemaField("target_bq_table", "STRING", mode="REQUIRED", description="Full path to master table"),
]

def recreate_config_table():
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(CONFIG_TABLE_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{CONFIG_TABLE_ID}"
    
    # Delete existing table
    print(f"Deleting table {table_id} if it exists...")
    client.delete_table(table_id, not_found_ok=True)
    
    # Create new table
    table = bigquery.Table(table_ref, schema=schema)
    try:
        client.create_table(table)
        print(f"Table {table.full_table_id} created with new schema.")
    except Exception as e:
        print(f"Table creation failed: {e}")
        raise e

if __name__ == "__main__":
    recreate_config_table()
