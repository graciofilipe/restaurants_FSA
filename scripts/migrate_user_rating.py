import os
from google.cloud import bigquery

def migrate(bq_path: str = os.environ.get("BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master")):
    client = bigquery.Client(project=bq_path.split(".")[0])
    try:
        client.query(f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS user_rating INT64").result()
    except Exception:
        pass

if __name__ == "__main__":
    migrate()
