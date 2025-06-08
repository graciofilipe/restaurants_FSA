import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
from bq_utils import read_from_bigquery, update_manual_review, BigQueryExecutionError # Ensure this import matches your file structure
from google.cloud import bigquery, exceptions # Import exceptions for error testing
# Attempt to import GenericGBQException for more specific error testing if available
try:
    from pandas_gbq.gbq import GenericGBQException
except ImportError:
    GenericGBQException = None # Fallback if pandas_gbq is not installed or structure differs

# --- Tests for read_from_bigquery (New implementation with pandas-gbq) ---

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_success(mock_read_gbq):
    """Test successful data retrieval with pandas_gbq.read_gbq."""
    expected_df = pd.DataFrame({'fhrsid': [123], 'data': ['test data']}) # Assuming fhrsid in DataFrame could also be int
    mock_read_gbq.return_value = expected_df

    fhrsid_list = [123, 456] # Changed to list of integers
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'arrayType': {'type': 'INT64'}}, # Changed to INT64
                    'parameterValue': {'arrayValues': [{'value': f_id} for f_id in fhrsid_list]}
                }
            ]
        }
    }

    df_result = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    mock_read_gbq.assert_called_once_with(
        expected_query,
        project_id=project_id,
        configuration=expected_configuration
    )
    pd.testing.assert_frame_equal(df_result, expected_df)

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_empty_result(mock_read_gbq):
    """Test retrieval of an empty DataFrame when no data is found."""
    mock_read_gbq.return_value = pd.DataFrame() # Empty DataFrame

    fhrsid_list = [789] # Changed to list of integers
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    df_result = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert df_result.empty
    mock_read_gbq.assert_called_once()

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_raises_bigqueryexecutionerror_on_generic_exception(mock_read_gbq):
    """Test that BigQueryExecutionError is raised for generic exceptions from read_gbq."""
    mock_read_gbq.side_effect = Exception("Simulated generic error from read_gbq")

    fhrsid_list = [101] # Changed to list of integers
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    with pytest.raises(BigQueryExecutionError) as excinfo:
        read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert "Simulated generic error from read_gbq" in str(excinfo.value)

if GenericGBQException: # Only run this test if GenericGBQException was successfully imported
    @patch('bq_utils.pandas_gbq.read_gbq')
    def test_read_from_bigquery_raises_bigqueryexecutionerror_on_genericgbqexception(mock_read_gbq):
        """Test that BigQueryExecutionError is raised for pandas_gbq.gbq.GenericGBQException."""
        mock_read_gbq.side_effect = GenericGBQException("Simulated GenericGBQException")

        fhrsid_list = [102] # Changed to list of integers
        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        with pytest.raises(BigQueryExecutionError) as excinfo:
            read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

        assert "Simulated GenericGBQException" in str(excinfo.value)

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_raises_bigqueryexecutionerror_on_googleclouderror(mock_read_gbq):
    """Test that BigQueryExecutionError is raised for google.cloud.exceptions.GoogleCloudError."""
    mock_read_gbq.side_effect = exceptions.GoogleCloudError("Simulated GoogleCloudError")

    fhrsid_list = [103] # Changed to list of integers
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    with pytest.raises(BigQueryExecutionError) as excinfo:
        read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert "Simulated GoogleCloudError" in str(excinfo.value)

# --- Tests for update_manual_review ---

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
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


# --- Tests for update_manual_review (Batch Operations) ---

@patch('bq_utils.st') # Mock Streamlit
@patch('bq_utils.bigquery.Client') # Mock BigQuery Client
def test_update_manual_review_batch_success(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.result.return_value = None # Simulate successful job completion

    fhrsid_list = ["101", "102", "103"]
    manual_review_value = "BatchApproved"
    project_id = "batch-proj"
    dataset_id = "batch-dset"
    table_id = "batch-tbl"

    result = update_manual_review(fhrsid_list, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.assert_called_once_with(project=project_id)

    expected_query = f"""
        UPDATE `{project_id}.{dataset_id}.{table_id}`
        SET manual_review = @manual_review_value
        WHERE fhrsid IN UNNEST(@fhrsid_list)
    """
    args, kwargs = mock_bq_client_instance.query.call_args
    actual_query = args[0]
    job_config = kwargs.get('job_config')

    assert "".join(actual_query.split()) == "".join(expected_query.split())
    assert job_config is not None

    expected_params = [
        bigquery.ScalarQueryParameter("manual_review_value", "STRING", manual_review_value),
        bigquery.ArrayQueryParameter("fhrsid_list", "STRING", fhrsid_list),
    ]

    assert len(job_config.query_parameters) == len(expected_params)
    # Check for scalar parameter
    scalar_param_actual = next(p for p in job_config.query_parameters if p.name == "manual_review_value")
    scalar_param_expected = next(p for p in expected_params if p.name == "manual_review_value")
    assert scalar_param_actual.name == scalar_param_expected.name
    assert scalar_param_actual.parameter_type.type_ == scalar_param_expected.parameter_type.type_
    assert scalar_param_actual.parameter_value.value == scalar_param_expected.parameter_value.value

    # Check for array parameter
    array_param_actual = next(p for p in job_config.query_parameters if p.name == "fhrsid_list")
    array_param_expected = next(p for p in expected_params if p.name == "fhrsid_list")
    assert array_param_actual.name == array_param_expected.name
    assert array_param_actual.parameter_type.array_type.type_ == array_param_expected.parameter_type.array_type.type_ # Check array element type
    assert array_param_actual.parameter_value.array_values[0].value == array_param_expected.parameter_value.array_values[0].value # Check first element as sample

    mock_query_job.result.assert_called_once()
    mock_st.success.assert_called_once_with(f"Successfully updated manual_review for FHRSIDs: {', '.join(fhrsid_list)} to '{manual_review_value}'.")
    assert result is True

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_update_manual_review_batch_bq_error(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.result.side_effect = exceptions.GoogleCloudError("Test BQ API Error on batch update")

    fhrsid_list = ["201", "202"]
    manual_review_value = "BatchRejected"
    project_id = "batch-proj"
    dataset_id = "batch-dset"
    table_id = "batch-tbl"

    result = update_manual_review(fhrsid_list, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.assert_called_once_with(project=project_id)
    mock_bq_client_instance.query.assert_called_once()
    mock_query_job.result.assert_called_once()
    # Check that the print statement for backend logging was also called
    mock_st.error.assert_called_once_with(f"Error updating manual_review for FHRSIDs: {', '.join(fhrsid_list)}: Test BQ API Error on batch update")
    assert result is False

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_update_manual_review_batch_empty_list(mock_bq_client_constructor, mock_st):
    fhrsid_list = []
    manual_review_value = "BatchEmpty"
    project_id = "batch-proj"
    dataset_id = "batch-dset"
    table_id = "batch-tbl"

    result = update_manual_review(fhrsid_list, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.return_value.query.assert_not_called() # Ensure query is not made
    mock_st.warning.assert_called_once_with("No FHRSIDs provided for update.")
    assert result is False

# (The old tests for read_from_bigquery that used bigquery.Client directly are removed by the SEARCH/REPLACE above)
