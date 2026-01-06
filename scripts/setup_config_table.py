from google.cloud import bigquery
import os

# Default Project ID - ideally this should be fetched from environment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants" 
CONFIG_TABLE_ID = "config_search_params"

# Schema definition
schema = [
    bigquery.SchemaField("coordinates", "STRING", mode="REQUIRED", description="Lat,Long pairs"),
    bigquery.SchemaField("max_results", "INTEGER", mode="REQUIRED", description="API fetch limit per coordinate"),
    bigquery.SchemaField("target_bq_table", "STRING", mode="REQUIRED", description="Full path to master table"),
]

def create_config_table():
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(CONFIG_TABLE_ID)
    
    table = bigquery.Table(table_ref, schema=schema)
    
    try:
        client.create_table(table)
        print(f"Table {table.full_table_id} created.")
        
        # Populate with initial data
        rows_to_insert = [
            {
                "coordinates": "51.5074,-0.1278", # London
                "max_results": 200,
                "target_bq_table": f"{PROJECT_ID}.{DATASET_ID}.fsa_master"
            }
        ]
        
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
            
    except Exception as e:
        print(f"Table creation failed (might already exist): {e}")

if __name__ == "__main__":
    create_config_table()
