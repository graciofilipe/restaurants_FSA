import argparse
import logging
import os
from typing import List, Optional
from google.cloud import bigquery
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POSTCODES_API_URL = "https://api.postcodes.io/postcodes"
DEFAULT_BQ_PATH = os.environ.get(
    "BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
)

def ensure_demographics_table(
    client: bigquery.Client, project_id: str, dataset_id: str, target_table: str
) -> bigquery.Table:
    """Ensures the target postcode demographics reference table exists."""
    table_ref = f"{project_id}.{dataset_id}.{target_table}"
    schema = [
        bigquery.SchemaField("postcode", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lsoa", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("msoa", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("imd_rank", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("admin_district", "STRING", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_ref, schema=schema)
    return client.create_table(table, exists_ok=True)

def enrich_postcodes(
    project_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    master_table: str = "fsa_master",
    target_table: str = "uk_postcode_demographics",
    batch_size: int = 100,
    limit: Optional[int] = None,
) -> int:
    """
    Enriches missing postcodes from fsa_master using api.postcodes.io
    and saves them to uk_postcode_demographics.
    """
    if not project_id or not dataset_id:
        parts = DEFAULT_BQ_PATH.split(".")
        project_id = project_id or parts[0]
        dataset_id = dataset_id or parts[1]

    client = bigquery.Client(project=project_id)

    # 1. Ensure target reference table exists
    ensure_demographics_table(client, project_id, dataset_id, target_table)

    master_table_ref = f"{project_id}.{dataset_id}.{master_table}"
    target_table_ref = f"{project_id}.{dataset_id}.{target_table}"

    # 2. Query distinct missing postcodes from fsa_master
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        SELECT DISTINCT PostCode AS postcode
        FROM `{master_table_ref}`
        WHERE PostCode IS NOT NULL AND TRIM(PostCode) != ''
          AND REPLACE(UPPER(PostCode), ' ', '') NOT IN (
              SELECT REPLACE(UPPER(postcode), ' ', '')
              FROM `{target_table_ref}`
              WHERE postcode IS NOT NULL
          )
        {limit_clause}
    """
    logger.info("Checking for missing postcodes in master table...")
    try:
        results = client.query(query).result()
        missing_postcodes = [row.postcode.strip() for row in results if row.postcode]
    except Exception as e:
        logger.error(f"Failed to query missing postcodes from BigQuery: {e}")
        return 0

    if not missing_postcodes:
        logger.info("No missing postcodes to enrich. All postcodes are up to date.")
        return 0

    logger.info(f"Found {len(missing_postcodes)} postcodes to enrich. Fetching from api.postcodes.io...")

    # 3. Batch fetch from api.postcodes.io
    rows_to_insert = []
    for i in range(0, len(missing_postcodes), batch_size):
        batch = missing_postcodes[i : i + batch_size]
        try:
            response = requests.post(POSTCODES_API_URL, json={"postcodes": batch})
            if response.status_code == 200:
                data = response.json()
                for item in data.get("result", []):
                    q = item.get("query")
                    r = item.get("result")
                    if r and isinstance(r, dict):
                        rows_to_insert.append({
                            "postcode": q,
                            "lsoa": r.get("lsoa"),
                            "msoa": r.get("msoa"),
                            "imd_rank": r.get("index_of_multiple_deprivation"),
                            "admin_district": r.get("admin_district"),
                        })
                    else:
                        rows_to_insert.append({
                            "postcode": q,
                            "lsoa": None,
                            "msoa": None,
                            "imd_rank": None,
                            "admin_district": None,
                        })
            else:
                logger.warning(
                    f"api.postcodes.io returned status code {response.status_code} for batch {i}"
                )
        except Exception as e:
            logger.error(f"Error fetching batch {i} from api.postcodes.io: {e}")

    # 4. Insert enriched rows into BigQuery target table
    if not rows_to_insert:
        logger.warning("No valid rows to insert.")
        return 0

    logger.info(f"Inserting {len(rows_to_insert)} enriched postcode rows into {target_table_ref}...")
    errors = client.insert_rows_json(target_table_ref, rows_to_insert)
    if errors:
        logger.error(f"Errors occurred while inserting rows: {errors}")
        return 0

    logger.info(f"Successfully enriched and stored {len(rows_to_insert)} postcodes.")
    return len(rows_to_insert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich UK Postcode Demographics in BigQuery")
    parser.add_argument("--project_id", default=None, help="GCP Project ID")
    parser.add_argument("--dataset_id", default=None, help="BigQuery Dataset ID")
    parser.add_argument("--master_table", default="fsa_master", help="Master restaurant table ID")
    parser.add_argument("--target_table", default="uk_postcode_demographics", help="Target demographics table ID")
    parser.add_argument("--limit", type=int, default=None, help="Max number of missing postcodes to enrich")
    args = parser.parse_args()

    enrich_postcodes(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        master_table=args.master_table,
        target_table=args.target_table,
        limit=args.limit,
    )
