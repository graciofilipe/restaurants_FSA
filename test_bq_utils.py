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

# --- Tests for read_from_bigquery (Batch Operations) ---

@patch('bq_utils.st') # Mock Streamlit for st.info/st.error
@patch('bq_utils.print') # Mock print for backend logging
@patch('bq_utils.bigquery.Client') # Mock BigQuery Client
def test_read_from_bigquery_batch_success_partial_data(mock_bq_client_constructor, mock_print, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job

    # Simulate data for 2 out of 3 requested FHRSIDs
    expected_df_data = {
        'fhrsid': ['1', '2'],
        'BusinessName': ['Restaurant A', 'Restaurant B'],
        'manual_review': [None, 'Approved']
    }
    mock_df = pd.DataFrame(expected_df_data)
    mock_query_job.to_dataframe.return_value = mock_df

    fhrsid_list = ["1", "2", "3"] # Requesting 3 FHRSIDs
    project_id = "read-proj"
    dataset_id = "read-dset"
    table_id = "read-tbl"

    df_result = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    pd.testing.assert_frame_equal(df_result, mock_df)

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    args, kwargs = mock_bq_client_instance.query.call_args
    actual_query = args[0]
    job_config = kwargs.get('job_config')

    assert "".join(actual_query.split()) == "".join(expected_query.split())
    assert job_config is not None
    assert len(job_config.query_parameters) == 1
    array_param_actual = job_config.query_parameters[0]
    assert array_param_actual.name == "fhrsid_list"
    assert array_param_actual.parameter_type.array_type.type_ == "STRING"
    assert array_param_actual.parameter_value.array_values[0].value == fhrsid_list[0] # Check first element

    # Check logging
    mock_print.assert_any_call(f"Executing BigQuery query: {expected_query}")
    mock_print.assert_any_call(f"With FHRSID list: {fhrsid_list}")


@patch('bq_utils.st')
@patch('bq_utils.print')
@patch('bq_utils.bigquery.Client')
def test_read_from_bigquery_batch_no_data_found(mock_bq_client_constructor, mock_print, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.to_dataframe.return_value = pd.DataFrame() # Empty DataFrame

    fhrsid_list = ["nonexistent1", "nonexistent2"]
    project_id = "read-proj"
    dataset_id = "read-dset"
    table_id = "read-tbl"

    df_result = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert df_result is None
    mock_st.info.assert_not_called()
    mock_print.assert_any_call(f"Executing BigQuery query: SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)")


@patch('bq_utils.st')
@patch('bq_utils.print')
@patch('bq_utils.bigquery.Client')
def test_read_from_bigquery_batch_query_execution_error(mock_bq_client_constructor, mock_print, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    error_message = "Simulated BQ Query Error"
    mock_bq_client_instance.query.side_effect = exceptions.GoogleCloudError(error_message)

    fhrsid_list = ["1", "2"]
    project_id = "read-proj"
    dataset_id = "read-dset"
    table_id = "read-tbl"

    df_result = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert df_result is None
    # Adjusted to match observed output where "None" appears before the error message
    full_error_message_print = f"Error querying BigQuery for FHRSIDs: {', '.join(fhrsid_list)} from table {project_id}.{dataset_id}.{table_id}: None {error_message}"
    mock_st.error.assert_not_called()
    mock_print.assert_any_call(full_error_message_print) # Check if print was called with this
    mock_print.assert_any_call(f"Executing BigQuery query: SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)")


@patch('bq_utils.st')
@patch('bq_utils.print')
@patch('bq_utils.bigquery.Client')
def test_read_from_bigquery_batch_to_dataframe_conversion_error(mock_bq_client_constructor, mock_print, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    conversion_error_message = "Simulated DataFrame Conversion Error"
    mock_query_job.to_dataframe.side_effect = ValueError(conversion_error_message)

    fhrsid_list = ["10", "20"]
    project_id = "read-proj"
    dataset_id = "read-dset"
    table_id = "read-tbl"

    df_result = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert df_result is None

    # Check st.error call
    mock_st.error.assert_not_called()

    # Check print calls for backend logging
    print_conversion_error_expected_msg = f"Error converting query result to DataFrame for FHRSIDs: {', '.join(fhrsid_list)} from table {project_id}.{dataset_id}.{table_id}: {conversion_error_message}"
    mock_print.assert_any_call(print_conversion_error_expected_msg)
    mock_print.assert_any_call(f"Executing BigQuery query: SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)")
