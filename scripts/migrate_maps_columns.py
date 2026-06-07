import os
from google.cloud import bigquery

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
BQ_PATH = os.environ.get("BQ_PATH", DEFAULT_BQ_PATH)

def migrate():
    project_id, dataset_id, table_id = BQ_PATH.split(".")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    queries = [
        f"ALTER TABLE `{table_ref}` ADD COLUMN IF NOT EXISTS price_level INT64",
        f"ALTER TABLE `{table_ref}` ADD COLUMN IF NOT EXISTS maps_rating FLOAT64",
        f"ALTER TABLE `{table_ref}` ADD COLUMN IF NOT EXISTS maps_reviews INT64"
    ]

    for q in queries:
        try:
            print(f"Executing: {q}")
            job = client.query(q)
            job.result()
            print("Success.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    migrate()
