import argparse
from google.cloud import bigquery

def migrate(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    query = f"""
    ALTER TABLE `{table_ref}`
    ADD COLUMN IF NOT EXISTS predicted_user_rating FLOAT64
    """

    print(f"Executing migration on {table_ref}...")
    try:
        client.query(query).result()
        print("Migration successful: added predicted_user_rating.")
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", default="filipegracio-ai-learning")
    parser.add_argument("--dataset_id", default="filipegracio_fsa_restaurants")
    parser.add_argument("--table_id", default="fsa_master")
    args = parser.parse_args()

    migrate(args.project_id, args.dataset_id, args.table_id)
