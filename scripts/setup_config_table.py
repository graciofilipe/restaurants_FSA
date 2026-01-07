from google.cloud import bigquery
import os
from typing import List, Dict, Any

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

def create_config_table(initial_coordinates: List[Dict[str, float]] = None):
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(CONFIG_TABLE_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{CONFIG_TABLE_ID}"
    
    # Check if table exists, if so delete to ensure schema update (for setup script this is aggressive but requested)
    # Or just try to create. The previous script caught exception.
    # The Plan says "Create scripts/update_bq_schema.py (or modify setup) to recreate/migrate the table."
    # I will stick to "recreate" logic here for consistency with the new schema.
    
    client.delete_table(table_id, not_found_ok=True)
    
    table = bigquery.Table(table_ref, schema=schema)
    
    try:
        client.create_table(table)
        print(f"Table {table.full_table_id} created.")
        
        # Default data if none provided
        if initial_coordinates is None:
            initial_coordinates = [
                {"lat": 51.5074, "lon": -0.1278} # London
            ]
            
        rows_to_insert = []
        for coord in initial_coordinates:
            rows_to_insert.append({
                "latitude": coord["lat"],
                "longitude": coord["lon"],
                "radius": None, # Default
                "max_results": 5000, # Updated default
                "target_bq_table": f"{PROJECT_ID}.{DATASET_ID}.fsa_master"
            })
        
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
            
    except Exception as e:
        print(f"Table creation/population failed: {e}")

if __name__ == "__main__":
    # User provided coordinates (Lon, Lat) -> mapped to lat/lon
    
    user_coords = [
        {"lon": -0.1197, "lat": 51.428},
        {"lon": -0.432, "lat": 51.648},
        {"lon": -0.085325, "lat": 51.482954},
        {"lon": -0.068429, "lat": 51.468681},
        {"lon": -0.105524, "lat": 51.429732},
        {"lon": -0.112850, "lat": 51.402007},
        {"lon": -0.169207, "lat": 51.400014},
        {"lon": -0.197718, "lat": 51.365842},
        {"lon": -0.239801, "lat": 51.390909},
        {"lon": -0.265550, "lat": 51.398463},
        {"lon": -0.240377, "lat": 51.408053},
        {"lon": -0.130875, "lat": 51.458639}
    ]

    create_config_table(initial_coordinates=user_coords)