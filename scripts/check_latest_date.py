from google.cloud import bigquery

project_id = "filipegracio-ai-learning"
dataset_id = "filipegracio_fsa_restaurants"
table_id = "fsa_master"

client = bigquery.Client(project=project_id)
query = f"SELECT MAX(first_seen) as latest_date FROM `{project_id}.{dataset_id}.{table_id}`"

try:
    results = client.query(query).result()
    for row in results:
        print(f"Latest first_seen date: {row.latest_date}")
except Exception as e:
    print(f"Error querying table: {e}")
