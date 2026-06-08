import pytest
from unittest.mock import patch, MagicMock
from scripts.train_bqml_model import train_model

@patch('scripts.train_bqml_model.bigquery.Client')
def test_train_model_async(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_query_job = MagicMock()
    mock_query_job.job_id = "test_job_id"
    mock_client.query.return_value = mock_query_job

    job_id = train_model(
        project_id="test_project",
        dataset_id="test_dataset",
        table_id="test_table",
        model_name="test_model",
        run_async=True
    )

    assert job_id == "test_job_id"
    assert mock_client.query.call_count == 2
    assert mock_query_job.result.call_count == 1

@patch('scripts.train_bqml_model.bigquery.Client')
def test_train_model_sync(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_query_job = MagicMock()
    mock_query_job.job_id = "test_job_id"
    mock_client.query.return_value = mock_query_job

    job_id = train_model(
        project_id="test_project",
        dataset_id="test_dataset",
        table_id="test_table",
        model_name="test_model",
        run_async=False
    )

    assert job_id == "test_job_id"
    assert mock_client.query.call_count == 2
    assert mock_query_job.result.call_count == 2
