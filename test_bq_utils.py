import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
from bq_utils import read_from_bigquery, update_manual_review, BigQueryExecutionError, write_to_bigquery, sanitize_column_name, load_all_data_from_bq # Ensure this import matches your file structure
from google.cloud import bigquery, exceptions # Import exceptions for error testing
from google.cloud import bigquery # Ensure full import as per instruction
from google.auth.exceptions import DefaultCredentialsError # Added import

# Attempt to import GenericGBQException for more specific error testing if available
try:
    from pandas_gbq.gbq import GenericGBQException
except ImportError:
    GenericGBQException = None # Fallback if pandas_gbq is not installed or structure differs

# --- Tests for load_all_data_from_bq ---

@patch('bq_utils.pandas_gbq.read_gbq')
def test_load_all_data_from_bq_success(mock_read_gbq):
    """Test successful data loading and conversion to list of dicts."""
    sample_data = {'col1': [1, 2], 'col2': ['a', 'b']}
    mock_df = pd.DataFrame(sample_data)
    mock_read_gbq.return_value = mock_df

    project_id = 'test-proj'
    dataset_id = 'test-dset'
    table_id = 'test-tbl'

    result = load_all_data_from_bq(project_id, dataset_id, table_id)
    expected_result = [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}]

    assert result == expected_result
    mock_read_gbq.assert_called_once_with(f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`", project_id=project_id)

@patch('bq_utils.pandas_gbq.read_gbq')
def test_load_all_data_from_bq_empty_table(mock_read_gbq):
    """Test loading from an empty table returns an empty list."""
    mock_df = pd.DataFrame()
    mock_read_gbq.return_value = mock_df

    project_id = 'test-proj'
    dataset_id = 'test-dset'
    table_id = 'empty-tbl'

    result = load_all_data_from_bq(project_id, dataset_id, table_id)

    assert result == []
    mock_read_gbq.assert_called_once_with(f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`", project_id=project_id)

@patch('bq_utils.pandas_gbq.read_gbq')
@patch('builtins.print') # Mock print to check error logging
def test_load_all_data_from_bq_pandas_gbq_exception(mock_print, mock_read_gbq):
    """Test that GenericGBQException is caught and returns an empty list."""
    project_id = 'test-proj'
    dataset_id = 'test-dset'
    table_id = 'gbq-exception-tbl'
    error_message = "Simulated pandas_gbq.gbq.GenericGBQException"
    mock_read_gbq.side_effect = GenericGBQException(error_message)

    result = load_all_data_from_bq(project_id, dataset_id, table_id)

    assert result == []
    mock_read_gbq.assert_called_once_with(f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`", project_id=project_id)
    # Check if error was printed (optional, but good for verifying logging)
    mock_print.assert_any_call(f"Error loading data from BigQuery table {project_id}.{dataset_id}.{table_id}: {error_message}")

@patch('bq_utils.pandas_gbq.read_gbq')
@patch('builtins.print')
def test_load_all_data_from_bq_google_auth_exception(mock_print, mock_read_gbq):
    """Test that DefaultCredentialsError is caught and returns an empty list."""
    project_id = 'test-proj'
    dataset_id = 'test-dset'
    table_id = 'auth-exception-tbl'
    error_message = "Simulated DefaultCredentialsError"
    mock_read_gbq.side_effect = DefaultCredentialsError(error_message)

    result = load_all_data_from_bq(project_id, dataset_id, table_id)

    assert result == []
    mock_read_gbq.assert_called_once_with(f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`", project_id=project_id)
    mock_print.assert_any_call(f"Error loading data from BigQuery table {project_id}.{dataset_id}.{table_id}: {error_message}")

@patch('bq_utils.pandas_gbq.read_gbq')
@patch('builtins.print')
def test_load_all_data_from_bq_generic_exception(mock_print, mock_read_gbq):
    """Test that a generic Exception is caught and returns an empty list."""
    project_id = 'test-proj'
    dataset_id = 'test-dset'
    table_id = 'generic-exception-tbl'
    error_message = "Simulated generic Exception"
    mock_read_gbq.side_effect = Exception(error_message)

    result = load_all_data_from_bq(project_id, dataset_id, table_id)

    assert result == []
    mock_read_gbq.assert_called_once_with(f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`", project_id=project_id)
    mock_print.assert_any_call(f"An unexpected error occurred while loading data from BigQuery table {project_id}.{dataset_id}.{table_id}: {error_message}")

# --- Tests for read_from_bigquery (New implementation with pandas-gbq) ---

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_success(mock_read_gbq):
    """Test successful data retrieval with pandas_gbq.read_gbq."""
    expected_df = pd.DataFrame({'fhrsid': [123], 'data': ['test data']}) # Assuming fhrsid in DataFrame could also be int
    mock_read_gbq.return_value = expected_df

    fhrsid_list = ['123', '456'] # Ensure this is a list of strings
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'type': 'ARRAY', 'arrayType': {'type': 'STRING'}}, # Correctly STRING
                    'parameterValue': {'arrayValues': [{'value': '123'}, {'value': '456'}]} # Explicitly string values
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
    """Test retrieval of an empty DataFrame when no data is found for non-empty input list."""
    mock_read_gbq.return_value = pd.DataFrame() # Empty DataFrame

    fhrsid_list = ['789'] # Ensure list of strings
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'type': 'ARRAY', 'arrayType': {'type': 'STRING'}},
                    'parameterValue': {'arrayValues': [{'value': '789'}]}
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
    assert df_result.empty

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_empty_input_list(mock_read_gbq):
    """Test behavior when the input fhrsid_list is empty."""
    mock_read_gbq.return_value = pd.DataFrame() # read_gbq would return empty for an empty IN clause essentially

    fhrsid_list = [] # Empty list
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'type': 'ARRAY', 'arrayType': {'type': 'STRING'}},
                    'parameterValue': {'arrayValues': []} # Expect empty arrayValues
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
    assert df_result.empty

@patch('bq_utils.pandas_gbq.read_gbq')
def test_read_from_bigquery_raises_bigqueryexecutionerror_on_generic_exception(mock_read_gbq):
    """Test that BigQueryExecutionError is raised for generic exceptions from read_gbq."""
    mock_read_gbq.side_effect = Exception("Simulated generic error from read_gbq")

    fhrsid_list = ['101'] # Ensure list of strings
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

        fhrsid_list = ['102'] # Ensure list of strings
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

    fhrsid_list = ['103'] # Ensure list of strings
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


# --- Tests for write_to_bigquery ---

@patch('bq_utils.st') # Mock streamlit for st.success/st.error
@patch('bq_utils.bigquery.Client')
def test_write_to_bigquery_newratingpending_conversion(mock_bq_client_constructor, mock_st):
    """
    Tests the write_to_bigquery function, focusing on the conversion
    of the 'NewRatingPending' column to boolean (or pd.NA).
    """
    # Mock BigQuery client and its methods
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_load_job = MagicMock()
    mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
    mock_load_job.result.return_value = None # Simulate successful job completion

    # Prepare test data
    original_col_name = 'NewRatingPending'
    sanitized_col_name = sanitize_column_name(original_col_name) # Should be 'newratingpending'

    data = {
        'FHRSID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'BusinessName': ['Cafe A', 'Cafe B', 'Cafe C', 'Cafe D', 'Cafe E', 'Cafe F', 'Cafe G', 'Cafe H', 'Cafe I', 'Cafe J', 'Cafe K', 'Cafe L'],
        original_col_name: ["true", "False", "TRUE", "false", "TrUe", "FaLsE", "other", "", None, pd.NA, " existing_true ", " existing_false "]
    }
    df = pd.DataFrame(data)

    columns_to_select = ['FHRSID', 'BusinessName', original_col_name]

    # Define bq_schema, ensuring NewRatingPending is BOOLEAN
    # The names in bq_schema should be the final sanitized names expected by BigQuery.
    bq_schema = [
        bigquery.SchemaField(sanitize_column_name('FHRSID'), 'INTEGER'),
        bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
        bigquery.SchemaField(sanitized_col_name, 'BOOLEAN')
    ]

    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    # Call the function
    # Ensure all mock objects and parameters are correctly passed
    result = write_to_bigquery(
        df=df,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        columns_to_select=columns_to_select,
        bq_schema=bq_schema
    )

    # Assertions
    assert result is True, "write_to_bigquery should return True on success"
    mock_st.success.assert_called_once() # Check if Streamlit success message was called

    # Check that load_table_from_dataframe was called
    mock_bq_client_instance.load_table_from_dataframe.assert_called_once()

    # Capture the DataFrame passed to load_table_from_dataframe
    loaded_df_call = mock_bq_client_instance.load_table_from_dataframe.call_args
    assert loaded_df_call is not None, "load_table_from_dataframe was not called with any arguments"
    loaded_df = loaded_df_call[0][0] # First argument of the first call

    # Assert the 'newratingpending' column content and type
    expected_values = [
        True, False, True, False, True, False,
        pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA # "other", "", None, pd.NA, " existing_true ", " existing_false " all become pd.NA
    ]

    assert sanitized_col_name in loaded_df.columns, f"Column '{sanitized_col_name}' not found in loaded DataFrame. Columns are: {loaded_df.columns}"

    loaded_series = loaded_df[sanitized_col_name]
    assert len(loaded_series) == len(expected_values), \
        f"Length mismatch for column '{sanitized_col_name}': Expected {len(expected_values)}, got {len(loaded_series)}"

    for i in range(len(expected_values)):
        expected_val = expected_values[i]
        actual_val = loaded_series.iloc[i]
        # Use pd.isna() for reliable NA comparison
        if pd.isna(expected_val):
            assert pd.isna(actual_val), f"Value at index {i} for column '{sanitized_col_name}' should be pd.NA but was '{actual_val}' (type: {type(actual_val)})"
        else:
            assert actual_val == expected_val, f"Value at index {i} for column '{sanitized_col_name}' was '{actual_val}' (type: {type(actual_val)}), expected '{expected_val}'"

    # Check the dtype of the column
    # After the specified conversion, a column with True, False, and pd.NA will have dtype 'object'.
    # If pandas evolves to use BooleanDtype more aggressively by default, this could be pd.BooleanDtype().
    assert loaded_series.dtype == object or isinstance(loaded_series.dtype, pd.BooleanDtype), \
        f"Expected dtype for '{sanitized_col_name}' to be 'object' or pandas BooleanDtype, but got {loaded_series.dtype}"

    # Verify job_config
    job_config_passed = loaded_df_call[1]['job_config'] # Second argument (kwargs) of the first call
    assert job_config_passed.schema == bq_schema
    assert job_config_passed.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE


@patch('bq_utils.st') # Mock streamlit
@patch('bq_utils.bigquery.Client')
def test_write_to_bigquery_includes_gemini_insights_in_schema(mock_bq_client_constructor, mock_st):
    """
    Tests that write_to_bigquery correctly processes a DataFrame and
    passes a schema to BigQuery that includes 'gemini_insights'.
    """
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_load_job = MagicMock()
    mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
    mock_load_job.result.return_value = None # Simulate successful job

    # Prepare test DataFrame
    data = {
        'FHRSID': [1],
        'BusinessName': ['Test Cafe'],
        'gemini_insights': [None], # New field
        'manual_review': ['reviewed'],
        'NewRatingPending': ['false'] # Existing field needed for full logic
    }
    df = pd.DataFrame(data)

    # Define the columns to select, including the new one
    columns_to_select = ['FHRSID', 'BusinessName', 'gemini_insights', 'manual_review', 'NewRatingPending']

    # Define the expected BQ schema that should be constructed by the calling code (e.g. st_app.py)
    # and passed to write_to_bigquery. This test ensures write_to_bigquery uses it.
    # Note: Names here are *after* sanitization.
    expected_bq_schema_passed_to_function = [
        bigquery.SchemaField(sanitize_column_name('FHRSID'), 'INTEGER'), # Assuming sanitize_column_name makes it lowercase
        bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('gemini_insights'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('manual_review'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('NewRatingPending'), 'BOOLEAN')
    ]

    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    write_to_bigquery(
        df,
        project_id,
        dataset_id,
        table_id,
        columns_to_select,
        expected_bq_schema_passed_to_function # This is the schema being tested
    )

    # Assert that load_table_from_dataframe was called
    mock_bq_client_instance.load_table_from_dataframe.assert_called_once()

    # Get the arguments passed to load_table_from_dataframe
    call_args = mock_bq_client_instance.load_table_from_dataframe.call_args
    loaded_df = call_args[0][0]
    job_config_passed_to_bq = call_args[1]['job_config']

    # 1. Check that the DataFrame passed to BQ has the sanitized 'gemini_insights' column
    assert 'gemini_insights' in loaded_df.columns # After sanitization by write_to_bigquery
    assert loaded_df['gemini_insights'].iloc[0] is None # Or pd.NA depending on how it's handled

    # 2. Check that the schema in job_config matches what we passed.
    # This confirms that write_to_bigquery is using the schema it received.
    assert job_config_passed_to_bq.schema == expected_bq_schema_passed_to_function

    # 3. Optionally, verify that 'gemini_insights' is in the job_config schema
    gemini_field_in_schema = next((field for field in job_config_passed_to_bq.schema if field.name == 'gemini_insights'), None)
    assert gemini_field_in_schema is not None, "'gemini_insights' field not found in BigQuery job_config schema"
    assert gemini_field_in_schema.field_type == 'STRING'
    assert gemini_field_in_schema.mode == 'NULLABLE'

    mock_st.success.assert_called_once()
