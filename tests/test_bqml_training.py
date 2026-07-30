import pytest
from unittest.mock import patch, MagicMock
from scripts.train_bqml_model import train_model

@patch("scripts.train_bqml_model.bigquery.Client")
def test_bqml_training_dry_run(mock_bq_client):
    mock_client = MagicMock()
    mock_bq_client.return_value = mock_client
    mock_job = MagicMock()
    mock_job.job_id = "dry_run_job_id"
    mock_client.query.return_value = mock_job

    job_id = train_model(
        project_id="filipegracio-ai-learning",
        dataset_id="filipegracio_fsa_restaurants",
        table_id="fsa_master",
        model_name="restaurant_preference_model",
        dry_run=True,
    )
    assert job_id is not None or mock_client.query.called

@patch("scripts.train_bqml_model.bigquery.Client")
def test_bqml_training_async(mock_bq_client):
    mock_client = MagicMock()
    mock_bq_client.return_value = mock_client
    mock_job = MagicMock()
    mock_job.job_id = "async_job_id"
    mock_client.query.return_value = mock_job

    job_id = train_model(
        project_id="filipegracio-ai-learning",
        dataset_id="filipegracio_fsa_restaurants",
        table_id="fsa_master",
        model_name="restaurant_preference_model",
        run_async=True,
    )
    assert job_id == "async_job_id"
