import os
from google.cloud import bigquery

def migrate(bq_path: str = os.environ.get("BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master")):
    client = bigquery.Client(project=bq_path.split(".")[0])
    cols = ["price_level INT64", "maps_rating FLOAT64", "maps_reviews INT64"]
    for c in cols:
        try:
            client.query(f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS {c}").result()
        except Exception:
            pass

if __name__ == "__main__":
    migrate()
