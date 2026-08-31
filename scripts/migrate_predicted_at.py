import os
import logging
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate(bq_path: str = os.environ.get("BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master")):
    project_id = bq_path.split(".")[0]
    client = bigquery.Client(project=project_id)
    
    logger.info(f"Adding predicted_at TIMESTAMP column to {bq_path} if not exists...")
    try:
        client.query(f"ALTER TABLE `{bq_path}` ADD COLUMN IF NOT EXISTS predicted_at TIMESTAMP").result()
        logger.info("Column predicted_at added or already exists.")
    except Exception as e:
        logger.warning(f"ALTER TABLE warning: {e}")
        
    logger.info("Populating predicted_at for existing predicted restaurants with CURRENT_TIMESTAMP()...")
    try:
        query_job = client.query(f"""
            UPDATE `{bq_path}`
            SET predicted_at = CURRENT_TIMESTAMP()
            WHERE predicted_user_rating IS NOT NULL AND predicted_at IS NULL
        """)
        query_job.result()
        logger.info(f"Updated {query_job.num_dml_affected_rows} rows with today's prediction timestamp.")
    except Exception as e:
        logger.error(f"Failed to populate predicted_at: {e}")

if __name__ == "__main__":
    migrate()
