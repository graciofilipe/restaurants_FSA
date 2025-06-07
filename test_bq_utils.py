import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
from bq_utils import read_from_bigquery, update_manual_review # Ensure this import matches your file structure
from google.cloud import bigquery, exceptions # Import exceptions for error testing

# Fixture for mock BigQuery client (from existing tests)
@pytest.fixture
def mock_bigquery_client_general():
    with patch('google.cloud.bigquery.Client') as mock_client_constructor:
        mock_client_instance = MagicMock()
        mock_client_constructor.return_value = mock_client_instance
        yield mock_client_instance

# Test for read_from_bigquery (from existing tests)
def test_read_from_bigquery_calls_to_dataframe(mock_bigquery_client_general):
    '''
    Test that read_from_bigquery successfully calls to_dataframe()
    which would have failed if db-dtypes was not present.
    '''
    # Use the general fixture
    mock_client_instance = mock_bigquery_client_general
    mock_query_job = MagicMock()
    mock_client_instance.query.return_value = mock_query_job
    mock_query_job.to_dataframe.return_value = pd.DataFrame({'some_column': [1, 2]})

    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"
    fhrsid_list = ["12345"]

    try:
        df = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        mock_client_instance.query.return_value.to_dataframe.assert_called_once()
    except ImportError:
        pytest.fail("ImportError was raised, db-dtypes might still be missing or not importable.")
    except Exception as e:
        pytest.fail(f"read_from_bigquery raised an unexpected exception: {e}")

# --- Tests for update_manual_review ---

@patch('bq_utils.st') # Mock Streamlit
@patch('bq_utils.bigquery.Client') # Mock BigQuery Client
def test_update_manual_review_success(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.result.return_value = None # Simulate successful job completion

    fhrsid = "123"
    manual_review_value = "Approved"
    project_id = "test-proj"
    dataset_id = "test-dset"
    table_id = "test-tbl"

    result = update_manual_review(fhrsid, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.assert_called_once_with(project=project_id)

    expected_query = f"""
        UPDATE `{project_id}.{dataset_id}.{table_id}`
        SET manual_review = @manual_review_value
        WHERE fhrsid = @fhrsid
    """

    # Check job_config and query parameters
    # We need to inspect the call_args for client.query
    args, kwargs = mock_bq_client_instance.query.call_args
    actual_query = args[0]
    job_config = kwargs.get('job_config')

    # Normalize whitespace for query comparison (optional, but good for robustness)
    assert "".join(actual_query.split()) == "".join(expected_query.split())

    assert job_config is not None
    expected_params = [
        bigquery.ScalarQueryParameter("manual_review_value", "STRING", manual_review_value),
        bigquery.ScalarQueryParameter("fhrsid", "STRING", fhrsid),
    ]

    # Check parameters (order might matter depending on implementation, or check as a set)
    assert len(job_config.query_parameters) == len(expected_params)
    for p_expected in expected_params:
        found = False
        for p_actual in job_config.query_parameters:
            if p_actual.name == p_expected.name and \
               p_actual.parameter_type.type_ == p_expected.parameter_type.type_ and \
               p_actual.parameter_value.value == p_expected.parameter_value.value:
                found = True
                break
        assert found, f"Expected query parameter {p_expected.name} not found or mismatch."

    mock_query_job.result.assert_called_once()
    mock_st.success.assert_called_once_with(f"Successfully updated manual_review for FHRSID {fhrsid} to '{manual_review_value}'.")
    assert result is True

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_update_manual_review_bq_error_on_result(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.result.side_effect = exceptions.GoogleCloudError("Test BQ API Error on result")

    fhrsid = "456"
    manual_review_value = "Rejected"
    project_id = "test-proj"
    dataset_id = "test-dset"
    table_id = "test-tbl"

    result = update_manual_review(fhrsid, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.assert_called_once_with(project=project_id)
    mock_bq_client_instance.query.assert_called_once() # Query was called
    mock_query_job.result.assert_called_once() # result() was called
    mock_st.error.assert_called_once_with(f"Error updating manual_review for FHRSID {fhrsid}: Test BQ API Error on result")
    assert result is False

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_update_manual_review_bq_error_on_query(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_bq_client_instance.query.side_effect = exceptions.GoogleCloudError("Test BQ API Error on query")

    fhrsid = "789"
    manual_review_value = "Pending"
    project_id = "test-proj"
    dataset_id = "test-dset"
    table_id = "test-tbl"

    result = update_manual_review(fhrsid, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.assert_called_once_with(project=project_id)
    mock_bq_client_instance.query.assert_called_once() # Query was called
    mock_st.error.assert_called_once_with(f"Error updating manual_review for FHRSID {fhrsid}: Test BQ API Error on query")
    assert result is False

# To run these tests, use pytest from your terminal in the project directory
# Ensure google-cloud-bigquery and streamlit are installed or properly mocked if not in test environment
# Example: pip install pytest google-cloud-bigquery streamlit
# Then run: pytest
# (Note: Streamlit is only used for st.success/st.error, which are mocked here)

# If you had a class structure:
# import unittest
# class TestUpdateManualReview(unittest.TestCase):
#     @patch('bq_utils.st')
#     @patch('bq_utils.bigquery.Client')
#     def test_update_manual_review_success(self, mock_bq_client_constructor, mock_st):
#         # ... same logic ...
#         pass # etc.
# if __name__ == '__main__':
#    unittest.main()
# But pytest style is fine as shown above.
