from app.services.bq_utils import upsert_agent_insight
import datetime
import os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants"
TABLE_ID = "restaurant_agent_insights"

def verify():
    print("Verifying upsert...")
    data = {
        "fhrsid": "999999_TEST",
        "raw_insight": "Test insight",
        "cuisine_type": "Test Cuisine",
        "review_count": 5,
        "average_rating": 5.0,
        "updated_at": datetime.datetime.now().isoformat()
    }
    success = upsert_agent_insight(PROJECT_ID, DATASET_ID, TABLE_ID, data)
    if success:
        print("Upsert reported success. Please check BigQuery for fhrsid '999999_TEST'.")
    else:
        print("Upsert failed.")

if __name__ == "__main__":
    verify()
