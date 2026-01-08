from google.cloud import bigquery
import os

# Default Project ID
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants"
TABLE_ID = "restaurant_agent_insights"

# Schema definition
schema = [
    bigquery.SchemaField("fhrsid", "STRING", mode="REQUIRED", description="Unique identifier for the restaurant"),
    bigquery.SchemaField("raw_insight", "STRING", mode="NULLABLE", description="Raw text response from the agent"),
    bigquery.SchemaField("cuisine_type", "STRING", mode="NULLABLE", description="Type of restaurant"),
    bigquery.SchemaField("review_count", "INTEGER", mode="NULLABLE", description="Number of reviews"),
    bigquery.SchemaField("average_rating", "FLOAT", mode="NULLABLE", description="Average rating"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED", description="Timestamp of the last update"),
]

def create_table():
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    table = bigquery.Table(table_ref, schema=schema)
    
    try:
        # Check if table exists
        try:
            client.get_table(table_id)
            print(f"Table {table_id} already exists.")
            # Verify schema matches (optional, but good practice)
        except Exception:
            print(f"Creating table {table_id}...")
            client.create_table(table)
            print(f"Table {table.full_table_id} created.")
            
    except Exception as e:
        print(f"Table creation failed: {e}")
        raise e

if __name__ == "__main__":
    create_table()
