import os
import argparse
import logging
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

def migrate_schema_and_data(bq_path: str = DEFAULT_BQ_PATH, dry_run: bool = False):
    """
    1. Adds `in_scope` (BOOLEAN) and `rating_source` (STRING) columns to the BigQuery table.
    2. Runs DML queries to categorize legacy manual_review records into the new workflow.
    """
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    
    logger.info(f"Target BigQuery Table: {bq_path}")
    
    # 1. Column additions
    ddl_statements = [
        f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS in_scope BOOLEAN",
        f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS rating_source STRING",
    ]
    
    for stmt in ddl_statements:
        logger.info(f"Executing DDL: {stmt}")
        if not dry_run:
            try:
                client.query(stmt).result()
            except Exception as e:
                logger.warning(f"DDL execution notice: {e}")
                
    # 2. Data Categorization DML Statements
    dml_statements = [
        ("Step 1: Mark legacy rejected non-restaurants as out-of-scope (in_scope = FALSE)", f"""
            UPDATE `{bq_path}`
            SET in_scope = FALSE
            WHERE manual_review = 'rejected'
              AND (
                LOWER(maps_types) LIKE '%bakery%'
                OR LOWER(maps_types) LIKE '%cafe%'
                OR LOWER(maps_types) LIKE '%supermarket%'
                OR LOWER(JSON_EXTRACT_SCALAR(gemini_insights_structured, '$.6_establishment_integrity_is_sit_down_restaurant')) = 'false'
              )
              AND LOWER(maps_types) NOT LIKE '%restaurant%'
        """),
        ("Step 2: Mark remaining legacy rejected restaurants as in-scope (in_scope = TRUE, user_rating unassigned)", f"""
            UPDATE `{bq_path}`
            SET in_scope = TRUE
            WHERE manual_review = 'rejected' AND in_scope IS NULL
        """),
        ("Step 3: Mark legacy accepted/approved records as in-scope (in_scope = TRUE)", f"""
            UPDATE `{bq_path}`
            SET in_scope = TRUE
            WHERE manual_review IN ('accepted', 'approved')
        """),
        ("Step 4: Mark legacy pending/not reviewed records as unprocessed (in_scope = NULL)", f"""
            UPDATE `{bq_path}`
            SET in_scope = NULL
            WHERE manual_review IN ('pending', 'not reviewed') OR manual_review IS NULL
        """),
        ("Step 5: Restore restaurants offering takeaway back to in-scope (in_scope = TRUE)", f"""
            UPDATE `{bq_path}`
            SET in_scope = TRUE
            WHERE in_scope = FALSE
              AND (
                LOWER(maps_types) LIKE '%restaurant%'
                OR LOWER(JSON_EXTRACT_SCALAR(gemini_insights_structured, '$.6_establishment_integrity_is_sit_down_restaurant')) = 'true'
              )
        """)
    ]
    
    for desc, query in dml_statements:
        logger.info(f"Running DML {desc}...")
        if dry_run:
            logger.info(f"[DRY RUN] Would execute:\n{query.strip()}\n")
        else:
            try:
                job = client.query(query)
                job.result()
                logger.info(f"Successfully executed DML. Rows affected: {job.num_dml_affected_rows}")
            except Exception as e:
                logger.error(f"Failed DML {desc}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate FSA Master table to in_scope score-driven workflow")
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH, help="Full BigQuery table path (project.dataset.table)")
    parser.add_argument("--dry-run", action="store_true", help="Print DML statements without executing them")
    args = parser.parse_args()
    
    migrate_schema_and_data(bq_path=args.bq_path, dry_run=args.dry_run)
