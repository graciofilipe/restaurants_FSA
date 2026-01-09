import pytest
from unittest.mock import patch, MagicMock, call, ANY
# The function upsert_agent_insight does not exist yet, so this import will fail during actual execution if I don't implement it.
# However, I'm writing the test first.
try:
    from app.services.bq_utils import upsert_agent_insight, load_specific_agent_insights
except ImportError:
    pass # To allow writing the file without immediate error in the agent loop

from google.cloud import bigquery
import datetime

@patch('google.cloud.bigquery.Client')
def test_load_specific_agent_insights_success(mock_client_cls):
    # Setup mock
    mock_client = mock_client_cls.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    
    # Mock row results
    mock_rows = [
        {"fhrsid": "1", "cuisine_type": "Italian", "review_count": 10, "average_rating": 4.0, "raw_insight": "Good", "updated_at": datetime.datetime.now(datetime.timezone.utc)},
        {"fhrsid": "2", "cuisine_type": "Indian", "review_count": 20, "average_rating": 4.5, "raw_insight": "Spice", "updated_at": datetime.datetime.now(datetime.timezone.utc)}
    ]
    
    # Convert dicts to mock objects that behave like BigQuery rows (attribute access)
    def make_row(d):
        row = MagicMock()
        row.get.side_effect = d.get
        row.__getitem__.side_effect = d.__getitem__
        for k, v in d.items():
            setattr(row, k, v)
        return row

    mock_query_job.result.return_value = [make_row(row) for row in mock_rows]

    # Test data
    project_id = "test-project"
    dataset_id = "test-dataset"
    fhrsids = ["1", "2"]

    # Execute
    result = load_specific_agent_insights(project_id, dataset_id, fhrsids)

    # Verify
    assert len(result) == 2
    assert result[0]["fhrsid"] == "1"
    assert result[1]["fhrsid"] == "2"
    
    # Verify query
    query = mock_client.query.call_args[0][0]
    assert "IN UNNEST" in query
    
    job_config = mock_client.query.call_args[1]['job_config']
    params = {p.name: p.value for p in job_config.query_parameters}
    assert params['fhrsids'] == fhrsids

@patch('google.cloud.bigquery.Client')
def test_upsert_agent_insight_success(mock_client_cls):
    # Setup mock
    mock_client = mock_client_cls.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    mock_query_job.result.return_value = None
    mock_query_job.errors = None

    # Test data
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"
    insight_data = {
        "fhrsid": "123456",
        "raw_insight": "It is a nice place.",
        "cuisine_type": "Italian",
        "review_count": 100,
        "average_rating": 4.5,
        "updated_at": datetime.datetime.now().isoformat()
    }

    # Execute
    result = upsert_agent_insight(project_id, dataset_id, table_id, insight_data)

    # Verify
    assert result is True
    mock_client.query.assert_called_once()
    
    # Check query content (basic check)
    query = mock_client.query.call_args[0][0]
    job_config = mock_client.query.call_args[1]['job_config']
    
    assert "MERGE" in query
    # Check if parameters are set correctly
    params = {p.name: p.value for p in job_config.query_parameters}
    assert params['fhrsid'] == "123456"
    assert params['cuisine_type'] == "Italian"

@patch('google.cloud.bigquery.Client')
def test_upsert_agent_insight_failure(mock_client_cls):
    # Setup mock failure
    mock_client = mock_client_cls.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    mock_query_job.result.side_effect = Exception("BQ Error")

    # Test data
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"
    insight_data = {
        "fhrsid": "123456",
        "updated_at": datetime.datetime.now().isoformat()
    }

    # Execute
    result = upsert_agent_insight(project_id, dataset_id, table_id, insight_data)

    # Verify
    assert result is False
