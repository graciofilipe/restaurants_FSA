import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, ANY # Added ANY
from bq_utils import read_from_bigquery, update_manual_review, BigQueryExecutionError, write_to_bigquery, sanitize_column_name, load_all_data_from_bq, append_to_bigquery # Ensure this import matches your file structure
from google.cloud import bigquery, exceptions # Import exceptions for error testing
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
    if GenericGBQException:
        mock_read_gbq.side_effect = GenericGBQException(error_message)
    else: # Fallback if GenericGBQException is not available
        mock_read_gbq.side_effect = Exception(error_message)


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
    expected_df = pd.DataFrame({'fhrsid': ["123"], 'data': ['test data']}) # FHRSID is string
    mock_read_gbq.return_value = expected_df

    fhrsid_list = ["123", "456"] # List of strings
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'type': 'ARRAY', 'arrayType': {'type': 'STRING'}}, # Changed to STRING
                    'parameterValue': {'arrayValues': [{'value': "123"}, {'value': "456"}]} # List of strings
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
    mock_read_gbq.return_value = pd.DataFrame()

    fhrsid_list = ["789"] # List of strings
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'type': 'ARRAY', 'arrayType': {'type': 'STRING'}}, # Changed to STRING
                    'parameterValue': {'arrayValues': [{'value': "789"}]} # List of strings
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
    mock_read_gbq.return_value = pd.DataFrame()

    fhrsid_list = []
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    expected_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE fhrsid IN UNNEST(@fhrsid_list)"
    expected_configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'type': 'ARRAY', 'arrayType': {'type': 'STRING'}}, # Changed to STRING
                    'parameterValue': {'arrayValues': []} # Remains empty, type change is key
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

    fhrsid_list = ["101"] # List of strings
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    with pytest.raises(BigQueryExecutionError) as excinfo:
        read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert "Simulated generic error from read_gbq" in str(excinfo.value)

if GenericGBQException:
    @patch('bq_utils.pandas_gbq.read_gbq')
    def test_read_from_bigquery_raises_bigqueryexecutionerror_on_genericgbqexception(mock_read_gbq):
        """Test that BigQueryExecutionError is raised for pandas_gbq.gbq.GenericGBQException."""
        mock_read_gbq.side_effect = GenericGBQException("Simulated GenericGBQException")

        fhrsid_list = ["102"] # List of strings
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

    fhrsid_list = ["103"] # List of strings
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    with pytest.raises(BigQueryExecutionError) as excinfo:
        read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)

    assert "Simulated GoogleCloudError" in str(excinfo.value)


# --- Tests for update_manual_review (Batch Operations) ---

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_update_manual_review_batch_success(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.result.return_value = None

    fhrsid_list = ["101", "102", "103"] # List of strings
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
        bigquery.ArrayQueryParameter("fhrsid_list", "STRING", fhrsid_list), # Changed to STRING
    ]

    # Ensure job_config.query_parameters is not None before checking its length
    assert job_config.query_parameters is not None, "job_config.query_parameters should not be None"
    assert len(job_config.query_parameters) == len(expected_params)

    scalar_param_actual = next(p for p in job_config.query_parameters if p.name == "manual_review_value")
    scalar_param_expected = next(p for p in expected_params if p.name == "manual_review_value")
    assert scalar_param_actual.name == scalar_param_expected.name
    assert scalar_param_actual.type_ == scalar_param_expected.type_ # Fixed: Access .type_ directly
    assert scalar_param_actual.value == scalar_param_expected.value # Fixed: Access .value directly

    array_param_actual = next(p for p in job_config.query_parameters if p.name == "fhrsid_list")
    array_param_expected = next(p for p in expected_params if p.name == "fhrsid_list")
    assert array_param_actual.name == array_param_expected.name
    assert array_param_actual.array_type == array_param_expected.array_type # Fixed: array_type is the string itself

    # Compare array values carefully
    # Fixed: Access .values directly on ArrayQueryParameter, and then .value for each item if they are Structs/Scalars
    # For simple array of strings, it's just array_param_actual.values
    actual_array_values = array_param_actual.values
    expected_array_values = array_param_expected.values
    assert actual_array_values == expected_array_values


    mock_query_job.result.assert_called_once()
    fhrsid_list_str = [str(f_id) for f_id in fhrsid_list] # Convert to strings for join
    mock_st.success.assert_called_once_with(f"Successfully updated manual_review for FHRSIDs: {', '.join(fhrsid_list_str)} to '{manual_review_value}'.")
    assert result is True

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_update_manual_review_batch_bq_error(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_bq_client_instance.query.return_value = mock_query_job
    mock_query_job.result.side_effect = exceptions.GoogleCloudError("Test BQ API Error on batch update")

    fhrsid_list = ["201", "202"] # List of strings
    manual_review_value = "BatchRejected"
    project_id = "batch-proj"
    dataset_id = "batch-dset"
    table_id = "batch-tbl"

    result = update_manual_review(fhrsid_list, manual_review_value, project_id, dataset_id, table_id)

    mock_bq_client_constructor.assert_called_once_with(project=project_id)
    mock_bq_client_instance.query.assert_called_once()
    mock_query_job.result.assert_called_once()

    # Using ANY for the exception part of the message for robustness
    mock_st.error.assert_called_once_with(ANY)
    actual_error_call_args = mock_st.error.call_args[0][0]
    # Check that the core parts of the message are present
    # Convert integer list to string list for join in assertion
    fhrsid_list_str = [str(f_id) for f_id in fhrsid_list]
    assert f"Error updating manual_review for FHRSIDs: {', '.join(fhrsid_list_str)}:" in actual_error_call_args
    assert "Test BQ API Error on batch update" in actual_error_call_args
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

    mock_bq_client_constructor.return_value.query.assert_not_called()
    mock_st.warning.assert_called_once_with("No FHRSIDs provided for update.")
    assert result is False


# --- Tests for write_to_bigquery ---

@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_write_to_bigquery_newratingpending_conversion_and_fhrsid_type(mock_bq_client_constructor, mock_st): # Renamed for clarity
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_load_job = MagicMock()
    mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
    mock_load_job.result.return_value = None

    original_col_name = 'NewRatingPending'
    sanitized_col_name = sanitize_column_name(original_col_name)
    sanitized_fhrsid_col = sanitize_column_name('FHRSID') # This will be 'fhrsid'

    data = {
        'FHRSID': ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], # FHRSID is string
        'BusinessName': ['Cafe A', 'Cafe B', 'Cafe C', 'Cafe D', 'Cafe E', 'Cafe F', 'Cafe G', 'Cafe H', 'Cafe I', 'Cafe J', 'Cafe K', 'Cafe L'],
        original_col_name: ["true", "False", "TRUE", "false", "TrUe", "FaLsE", "other", "", None, pd.NA, " existing_true ", " existing_false "]
    }
    df = pd.DataFrame(data)
    # Ensure FHRSID column is string type
    df['FHRSID'] = df['FHRSID'].astype(str)

    columns_to_select = ['FHRSID', 'BusinessName', original_col_name]
    # Schema now expects STRING for fhrsid
    bq_schema = [
        bigquery.SchemaField(sanitized_fhrsid_col, 'STRING'),
        bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
        bigquery.SchemaField(sanitized_col_name, 'BOOLEAN')
    ]
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    result = write_to_bigquery(
        df=df.copy(), # Pass a copy to avoid modification issues in test
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        columns_to_select=columns_to_select,
        bq_schema=bq_schema
    )

    assert result is True, "write_to_bigquery should return True on success"
    mock_st.success.assert_called_once()
    mock_bq_client_instance.load_table_from_dataframe.assert_called_once()
    loaded_df_call = mock_bq_client_instance.load_table_from_dataframe.call_args
    assert loaded_df_call is not None, "load_table_from_dataframe was not called with any arguments"
    loaded_df = loaded_df_call[0][0]

    # Check FHRSID type after sanitization (should be 'fhrsid')
    # The key in loaded_df.columns will be the sanitized version, e.g., 'fhrsid'
    actual_sanitized_fhrsid_col_in_df = sanitize_column_name('FHRSID')
    assert actual_sanitized_fhrsid_col_in_df in loaded_df.columns, f"Column '{actual_sanitized_fhrsid_col_in_df}' not found. Columns: {loaded_df.columns}"
    assert pd.api.types.is_string_dtype(loaded_df[actual_sanitized_fhrsid_col_in_df]), \
        f"FHRSID column '{actual_sanitized_fhrsid_col_in_df}' in loaded_df should be string, got {loaded_df[actual_sanitized_fhrsid_col_in_df].dtype}"

    expected_newratingpending_values = [
        True, False, True, False, True, False,
        pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    ]
    assert sanitized_col_name in loaded_df.columns, f"Column '{sanitized_col_name}' not found in loaded DataFrame. Columns are: {loaded_df.columns}"
    loaded_newratingpending_series = loaded_df[sanitized_col_name]
    assert len(loaded_newratingpending_series) == len(expected_newratingpending_values), \
        f"Length mismatch for column '{sanitized_col_name}': Expected {len(expected_newratingpending_values)}, got {len(loaded_newratingpending_series)}"

    for i in range(len(expected_newratingpending_values)):
        expected_val = expected_newratingpending_values[i]
        actual_val = loaded_newratingpending_series.iloc[i]
        if pd.isna(expected_val):
            assert pd.isna(actual_val), f"Value at index {i} for column '{sanitized_col_name}' should be pd.NA but was '{actual_val}' (type: {type(actual_val)})"
        else:
            assert actual_val == expected_val, f"Value at index {i} for column '{sanitized_col_name}' was '{actual_val}' (type: {type(actual_val)}), expected '{expected_val}'"

    assert loaded_newratingpending_series.dtype == object or isinstance(loaded_newratingpending_series.dtype, pd.BooleanDtype), \
        f"Expected dtype for '{sanitized_col_name}' to be 'object' or pandas BooleanDtype, but got {loaded_newratingpending_series.dtype}"

    job_config_passed = loaded_df_call[1]['job_config']
    assert job_config_passed.schema == bq_schema
    assert job_config_passed.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE


@patch('bq_utils.st')
@patch('bq_utils.bigquery.Client')
def test_write_to_bigquery_includes_gemini_insights_in_schema(mock_bq_client_constructor, mock_st):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_load_job = MagicMock()
    mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
    mock_load_job.result.return_value = None

    data = {
        'FHRSID': ["1"], # FHRSID is string
        'BusinessName': ['Test Cafe'],
        'gemini_insights': [None],
        'manual_review': ['reviewed'],
        'NewRatingPending': ['false']
    }
    df = pd.DataFrame(data)
    df['FHRSID'] = df['FHRSID'].astype(str) # Ensure string type
    columns_to_select = ['FHRSID', 'BusinessName', 'gemini_insights', 'manual_review', 'NewRatingPending']
    expected_bq_schema_passed_to_function = [
        bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING'), # FHRSID is STRING
        bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('gemini_insights'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('manual_review'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('NewRatingPending'), 'BOOLEAN')
    ]
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    write_to_bigquery(
        df.copy(), # Pass a copy
        project_id,
        dataset_id,
        table_id,
        columns_to_select,
        expected_bq_schema_passed_to_function
    )
    mock_bq_client_instance.load_table_from_dataframe.assert_called_once()
    call_args = mock_bq_client_instance.load_table_from_dataframe.call_args
    loaded_df = call_args[0][0]
    job_config_passed_to_bq = call_args[1]['job_config']

    assert 'gemini_insights' in loaded_df.columns
    assert loaded_df['gemini_insights'].iloc[0] is None
    assert job_config_passed_to_bq.schema == expected_bq_schema_passed_to_function
    gemini_field_in_schema = next((field for field in job_config_passed_to_bq.schema if field.name == 'gemini_insights'), None)
    assert gemini_field_in_schema is not None, "'gemini_insights' field not found in BigQuery job_config schema"
    assert gemini_field_in_schema.field_type == 'STRING'
    assert gemini_field_in_schema.mode == 'NULLABLE'
    mock_st.success.assert_called_once()


# --- Tests for append_to_bigquery ---
import unittest

class TestAppendToBigQuery(unittest.TestCase): # Changed to use unittest.TestCase for easier class-based structure
    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_append_successful(self, mock_bq_client, mock_st):
        mock_job = MagicMock()
        mock_bq_client.return_value.load_table_from_dataframe.return_value = mock_job
        mock_job.result.return_value = None

        data = {
            'fhrsid': ["1", "2"], # fhrsid is string
            'businessname': ['Restaurant A', 'Restaurant B'],
            'newratingpending': ['false', 'true'],
            'geocode_latitude': ['51.0', '52.0'],
            'geocode_longitude': ['-0.1', '-0.2']
        }
        df = pd.DataFrame(data)
        df['fhrsid'] = df['fhrsid'].astype(str) # Ensure string type

        # append_to_bigquery expects df column names to be already sanitized
        # and matching the schema field names.
        df.columns = [sanitize_column_name(col) for col in df.columns]


        schema = [
            bigquery.SchemaField(sanitize_column_name('fhrsid'), 'STRING'), # fhrsid is STRING
            bigquery.SchemaField(sanitize_column_name('businessname'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('newratingpending'), 'BOOLEAN'),
            bigquery.SchemaField(sanitize_column_name('geocode_latitude'), 'FLOAT'),
            bigquery.SchemaField(sanitize_column_name('geocode_longitude'), 'FLOAT'),
        ]

        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        # Pass a copy of df to the function
        result = append_to_bigquery(df.copy(), project_id, dataset_id, table_id, schema)

        self.assertTrue(result)
        mock_bq_client.return_value.load_table_from_dataframe.assert_called_once()

        # Correctly access call_args for an instance method mock
        args_list = mock_bq_client.return_value.load_table_from_dataframe.call_args_list
        self.assertEqual(len(args_list), 1) # Ensure it was called once

        called_df = args_list[0][0][0] # First arg of first call
        # called_table_ref = args_list[0][0][1] # Second arg of first call
        job_config = args_list[0][1]['job_config'] # job_config from kwargs of first call

        self.assertEqual(job_config.write_disposition, bigquery.WriteDisposition.WRITE_APPEND)
        self.assertEqual(job_config.schema, schema)

        # Assert fhrsid is string type in the DataFrame passed to BQ
        self.assertTrue(pd.api.types.is_string_dtype(called_df[sanitize_column_name('fhrsid')]))
        self.assertTrue(pd.api.types.is_bool_dtype(called_df[sanitize_column_name('newratingpending')]))
        self.assertTrue(pd.api.types.is_numeric_dtype(called_df[sanitize_column_name('geocode_latitude')]))
        self.assertTrue(pd.api.types.is_numeric_dtype(called_df[sanitize_column_name('geocode_longitude')]))
        mock_st.success.assert_called_once()

    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_append_failure_on_load(self, mock_bq_client, mock_st):
        mock_bq_client.return_value.load_table_from_dataframe.side_effect = Exception("BQ API error")

        df = pd.DataFrame({'col1': [1]})
        df.columns = [sanitize_column_name(col) for col in df.columns] # Sanitize
        schema = [bigquery.SchemaField(sanitize_column_name('col1'), 'INTEGER')]

        result = append_to_bigquery(df.copy(), "p", "d", "t", schema) # Pass a copy
        self.assertFalse(result)
        mock_st.error.assert_called_once()

    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_append_empty_dataframe(self, mock_bq_client, mock_st):
        """Test that append_to_bigquery handles an empty DataFrame correctly."""
        # This test assumes that the function might try to process an empty df,
        # or that an empty df might result from subsetting if schema doesn't match.
        # append_to_bigquery's current logic for df_subset = df[schema_columns].copy()
        # would raise a KeyError if schema_columns are not in df.
        # For this test, let's assume df is empty but has columns matching schema.

        empty_df = pd.DataFrame(columns=[sanitize_column_name('col1'), sanitize_column_name('col2')])
        schema = [
            bigquery.SchemaField(sanitize_column_name('col1'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('col2'), 'INTEGER'),
        ]

        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        result = append_to_bigquery(empty_df.copy(), project_id, dataset_id, table_id, schema)

        # If the DataFrame is empty, load_table_from_dataframe might not be called,
        # or it might be called and BQ handles empty loads.
        # Let's assume for now it's a success if it doesn't error and BQ is called.
        # If BQ is not meant to be called with an empty df, the test needs adjustment.
        self.assertTrue(result) # Or assert based on expected behavior for empty df
        mock_bq_client.return_value.load_table_from_dataframe.assert_called_once()
        mock_st.success.assert_called_once() # Assuming BQ client handles empty load as success

    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_column_subsetting_based_on_schema(self, mock_bq_client, mock_st):
        mock_job = MagicMock()
        mock_bq_client.return_value.load_table_from_dataframe.return_value = mock_job
        mock_job.result.return_value = None

        data = {
            sanitize_column_name('id'): [1],
            sanitize_column_name('name'): ['Alice'],
            sanitize_column_name('extra_field'): ['skip_me'] # This field is not in schema
        }
        df = pd.DataFrame(data)

        # Schema only includes 'id' and 'name'
        schema = [
            bigquery.SchemaField(sanitize_column_name('id'), 'INTEGER'),
            bigquery.SchemaField(sanitize_column_name('name'), 'STRING'),
        ]

        append_to_bigquery(df.copy(), "p", "d", "t", schema)

        called_df = mock_bq_client.return_value.load_table_from_dataframe.call_args[0][0]

        # Assert that 'extra_field' is NOT in the DataFrame passed to BQ
        self.assertNotIn(sanitize_column_name('extra_field'), called_df.columns)
        self.assertIn(sanitize_column_name('id'), called_df.columns)
        self.assertIn(sanitize_column_name('name'), called_df.columns)
        self.assertEqual(len(called_df.columns), 2)

    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_append_to_bigquery_fhrsid_is_string(self, mock_bq_client_constructor, mock_st): # Existing test, modified
        """Test fhrsid column is handled as string for BQ load when schema is STRING, including non-string inputs."""
        mock_bq_client_instance = mock_bq_client_constructor.return_value
        mock_load_job = MagicMock()
        mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
        mock_load_job.result.return_value = None

        sample_data = {
            'fhrsid': [123, "456", 789, None, pd.NA, 10.5],
            'another_col': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        }
        sample_df = pd.DataFrame(sample_data)

        bq_schema = [
            bigquery.SchemaField('fhrsid', 'STRING'),
            bigquery.SchemaField('another_col', 'STRING')
        ]
        project_id, dataset_id, table_id = "p", "d", "t"

        result = append_to_bigquery(sample_df.copy(), project_id, dataset_id, table_id, bq_schema)

        self.assertTrue(result, "append_to_bigquery should return True on success")
        mock_st.success.assert_called_once()

        mock_bq_client_instance.load_table_from_dataframe.assert_called_once()
        args, kwargs = mock_bq_client_instance.load_table_from_dataframe.call_args
        loaded_df = args[0]

        self.assertTrue(pd.api.types.is_string_dtype(loaded_df['fhrsid']) or loaded_df['fhrsid'].dtype == 'object',
                        f"fhrsid column should be string type or object, but was {loaded_df['fhrsid'].dtype}")

        expected_str_values = ["123", "456", "789", "None", "<NA>", "10.5"]
        self.assertEqual(loaded_df['fhrsid'].tolist(), expected_str_values)

    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_append_to_bigquery_fhrsid_is_integer_and_coerces_invalid(self, mock_bq_client_constructor, mock_st): # New test
        """Test fhrsid is numeric for INTEGER schema, and invalid strings become NaN."""
        mock_bq_client_instance = mock_bq_client_constructor.return_value
        mock_load_job = MagicMock()
        mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
        mock_load_job.result.return_value = None

        sample_data = {
            'fhrsid': ["123", 456, "789", "invalid_id", None],
            'another_col': ['value1', 'value2', 'value3', 'value4', 'value5']
        }
        sample_df = pd.DataFrame(sample_data)

        bq_schema = [
            bigquery.SchemaField('fhrsid', 'INTEGER'),
            bigquery.SchemaField('another_col', 'STRING')
        ]
        project_id, dataset_id, table_id = "p", "d", "t"

        result = append_to_bigquery(sample_df.copy(), project_id, dataset_id, table_id, bq_schema)

        self.assertTrue(result, "append_to_bigquery should return True")
        mock_st.success.assert_called_once()

        mock_bq_client_instance.load_table_from_dataframe.assert_called_once()
        args, kwargs = mock_bq_client_instance.load_table_from_dataframe.call_args
        loaded_df = args[0]

        self.assertTrue(pd.api.types.is_numeric_dtype(loaded_df['fhrsid']),
                        f"fhrsid column should be numeric, but was {loaded_df['fhrsid'].dtype}")

        expected_values = [123.0, 456.0, 789.0, np.nan, np.nan]
        expected_series = pd.Series(expected_values, name='fhrsid', dtype=loaded_df['fhrsid'].dtype)
        # Using check_dtype=False as type might change from object to float64 due to NaNs
        pd.testing.assert_series_equal(loaded_df['fhrsid'], expected_series, check_dtype=False)

    @patch('bq_utils.st')
    @patch('bq_utils.bigquery.Client')
    def test_append_to_bigquery_fhrsid_conversion_error_to_nan_for_int64(self, mock_bq_client_constructor, mock_st): # New test
        """Test fhrsid converts to NaN for unparseable strings when BQ schema is INT64."""
        mock_bq_client_instance = mock_bq_client_constructor.return_value
        mock_load_job = MagicMock()
        mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
        mock_load_job.result.return_value = None

        sample_data = {
            'fhrsid': ["123", "abc", "456", None, pd.NA],
            'another_col': ['value1', 'value2', 'value3', 'value4', 'value5']
        }
        sample_df = pd.DataFrame(sample_data)

        bq_schema = [
            bigquery.SchemaField('fhrsid', 'INT64'),
            bigquery.SchemaField('another_col', 'STRING')
        ]
        project_id, dataset_id, table_id = "p", "d", "t"

        result = append_to_bigquery(sample_df.copy(), project_id, dataset_id, table_id, bq_schema)

        self.assertTrue(result, "append_to_bigquery should return True")
        mock_st.success.assert_called_once()
        mock_bq_client_instance.load_table_from_dataframe.assert_called_once()

        args, kwargs = mock_bq_client_instance.load_table_from_dataframe.call_args
        loaded_df = args[0]

        self.assertTrue(pd.api.types.is_numeric_dtype(loaded_df['fhrsid']),
                        f"fhrsid column should be numeric after coercion, but was {loaded_df['fhrsid'].dtype}")

        expected_values = [123.0, np.nan, 456.0, np.nan, np.nan]
        expected_series = pd.Series(expected_values, name='fhrsid', dtype=loaded_df['fhrsid'].dtype)
        # Using check_dtype=False as type might change from object to float64 due to NaNs
        pd.testing.assert_series_equal(loaded_df['fhrsid'], expected_series, check_dtype=False)


# If __name__ == '__main__':
#     unittest.main() # This allows running file directly if not using pytest
# For pytest, this is not strictly necessary.
# To run with pytest, ensure pytest and necessary libraries are installed:
# pip install pytest pandas google-cloud-bigquery streamlit
# Then, from the terminal in the directory containing this file:
# pytest test_bq_utils.py
# Or simply: pytest (if it's in a standard test discovery path)
