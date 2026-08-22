import pytest
from google.cloud import bigquery
from scripts.train_bqml_model import train_model

def test_bqml_training_with_bad_json():
    # We will test the query generation by executing a select statement derived from the script
    client = bigquery.Client()
    
    query_template = f"""
    SELECT
      IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.1_value_and_volume_rating') AS INT64), 0) AS score_1_value_and_volume_rating
    FROM UNNEST([
      STRUCT('{{\"1_value_and_volume_rating\": 5}}' AS gemini_insights_structured),
      STRUCT('{{\"1_value_and_volume_rating\": \"N/A\"}}' AS gemini_insights_structured)
    ])
    """
    
    try:
        results = list(client.query(query_template).result())
        pytest.fail("Query should have failed due to CAST error, but it succeeded!")
    except Exception as e:
        assert "Bad int64 value" in str(e)

def test_bqml_training_with_safe_cast():
    client = bigquery.Client()
    query_template = f"""
    SELECT
      IFNULL(SAFE_CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.1_value_and_volume_rating') AS INT64), 0) AS score_1_value_and_volume_rating
    FROM UNNEST([
      STRUCT('{{\"1_value_and_volume_rating\": 5}}' AS gemini_insights_structured),
      STRUCT('{{\"1_value_and_volume_rating\": \"N/A\"}}' AS gemini_insights_structured)
    ])
    """
    
    try:
        results = list(client.query(query_template).result())
        assert results[0].score_1_value_and_volume_rating == 5
        assert results[1].score_1_value_and_volume_rating == 0
    except Exception as e:
        pytest.fail(f"SAFE_CAST query failed: {e}")
