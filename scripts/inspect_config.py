from google.cloud import bigquery
import pandas as pd

project_id = "filipegracio-ai-learning"
dataset_id = "filipegracio_fsa_restaurants"
table_id = "config_search_params"

client = bigquery.Client(project=project_id)
query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"

try:
    df = client.query(query).to_dataframe()
    if df.empty:
        print("Config table is empty.")
    else:
        print("Current Configuration:")
        print(df.to_string(index=False))
except Exception as e:
    print(f"Error querying table: {e}")
