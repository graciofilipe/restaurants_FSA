import os
from google.cloud import bigquery

def main(bq_path: str = os.environ.get("BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master")):
    client = bigquery.Client(project=bq_path.split(".")[0])
    cols = ["latitude FLOAT64", "longitude FLOAT64", "maps_url STRING", "business_status STRING", "website_url STRING", "maps_types STRING"]
    for c in cols:
        try:
            client.query(f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS {c}").result()
        except Exception:
            pass

if __name__ == "__main__":
    main()
