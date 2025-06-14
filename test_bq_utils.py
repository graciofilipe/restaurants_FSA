import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, ANY # Added ANY
from bq_utils import (
    read_from_bigquery, update_manual_review, BigQueryExecutionError,
    write_to_bigquery, sanitize_column_name, load_all_data_from_bq,
    append_to_bigquery, ORIGINAL_COLUMNS_TO_KEEP, NEW_BQ_SCHEMA
)
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

@patch('builtins.print') # Patched print
@patch('bq_utils.bigquery.Client')
def test_write_to_bigquery_logic_with_fixed_schema(mock_bq_client_constructor, mock_print):
    mock_bq_client_instance = mock_bq_client_constructor.return_value
    mock_load_job = MagicMock()
    mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
    mock_load_job.result.return_value = None

    # Prepare input data according to ORIGINAL_COLUMNS_TO_KEEP
    data = {
        'FHRSID': ["1", "2"],
        'BusinessName': ['Cafe Foo', 'Bar Boo'],
        'AddressLine1': ['1 Street', '2 Avenue'],
        'AddressLine2': ['Town', 'City'],
        'AddressLine3': ['', ''],
        'PostCode': ['PC1 1PC', 'PC2 2PC'],
        'LocalAuthorityName': ['Council A', 'Council B'],
        'RatingValue': ['5', '3'], # API often gives strings
        'NewRatingPending': ['false', 'true'], # API often gives strings for boolean
        'first_seen': ['2023-01-01', '2023-01-02'],
        'manual_review': ['reviewed', 'not reviewed'],
        'gemini_insights': [None, 'Some insight']
    }
    # Ensure all columns from ORIGINAL_COLUMNS_TO_KEEP are present for robust test
    for col in ORIGINAL_COLUMNS_TO_KEEP:
        if col not in data:
            data[col] = [None] * len(data['FHRSID'])

    df = pd.DataFrame(data)
    # Ensure FHRSID is string as it would be from data_processing layer
    df['FHRSID'] = df['FHRSID'].astype(str)
    # NewRatingPending is also often a string from APIs
    df['NewRatingPending'] = df['NewRatingPending'].astype(str)


    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"

    # Call write_to_bigquery (no columns_to_select or bq_schema arguments)
    result = write_to_bigquery(
        df=df.copy(),
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )

    assert result is True, "write_to_bigquery should return True on success"
    # mock_st.success.assert_called_once() # st is not mocked here, print is.
    mock_bq_client_instance.load_table_from_dataframe.assert_called_once()

    loaded_df_call_args = mock_bq_client_instance.load_table_from_dataframe.call_args
    assert loaded_df_call_args is not None, "load_table_from_dataframe was not called"

    loaded_df = loaded_df_call_args[0][0]
    job_config_passed = loaded_df_call_args[1]['job_config']

    # Assertions on loaded_df (sanitized column names, types)
    sanitized_original_cols = sorted([sanitize_column_name(col) for col in ORIGINAL_COLUMNS_TO_KEEP])
    assert sorted(list(loaded_df.columns)) == sanitized_original_cols

    # Check FHRSID type (should be string as per NEW_BQ_SCHEMA)
    sanitized_fhrsid_col = sanitize_column_name('FHRSID')
    assert pd.api.types.is_string_dtype(loaded_df[sanitized_fhrsid_col]), \
        f"{sanitized_fhrsid_col} column in loaded_df should be string, got {loaded_df[sanitized_fhrsid_col].dtype}"
    assert loaded_df[sanitized_fhrsid_col].tolist() == ["1", "2"]

    # Check NewRatingPending type (should be boolean after processing)
    sanitized_nrp_col = sanitize_column_name('NewRatingPending')
    assert pd.api.types.is_bool_dtype(loaded_df[sanitized_nrp_col]) or loaded_df[sanitized_nrp_col].dtype == 'boolean', \
        f"{sanitized_nrp_col} column should be boolean, got {loaded_df[sanitized_nrp_col].dtype}"
    # Using .fillna(False) because pd.NA might be actual value for boolean if original was None/empty
    # and direct list comparison might fail. Expected: [False, True]
    # Handle potential pd.NA if conversion from None/"other" string results in it
    expected_bool_series = pd.Series([False, True], name=sanitized_nrp_col).astype('boolean')
    pd.testing.assert_series_equal(loaded_df[sanitized_nrp_col].astype('boolean'), expected_bool_series, check_dtype=False)


    # Assertions on job_config
    assert job_config_passed.schema == NEW_BQ_SCHEMA
    assert job_config_passed.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE

# Removed test_write_to_bigquery_includes_gemini_insights_in_schema as its logic is merged above.
# The new test test_write_to_bigquery_logic_with_fixed_schema covers all columns and schema.


# --- Tests for append_to_bigquery ---
import unittest

class TestAppendToBigQuery(unittest.TestCase): # Changed to use unittest.TestCase for easier class-based structure
    @patch('builtins.print') # Patched print
    @patch('bq_utils.bigquery.Client')
    def test_append_successful(self, mock_bq_client, mock_print):
        mock_job = MagicMock()
        mock_bq_client.return_value.load_table_from_dataframe.return_value = mock_job
        mock_job.result.return_value = None

        # Prepare data according to NEW_BQ_SCHEMA (sanitized names)
        sanitized_schema_names = [field.name for field in NEW_BQ_SCHEMA]
        data = {
            'fhrsid': ["1", "2"],
            'businessname': ['Restaurant A', 'Restaurant B'],
            'addressline1': ['Main St', 'High St'],
            'addressline2': ['Anytown', 'Othertown'],
            'addressline3': ['', ''],
            'postcode': ['A1 1AA', 'B2 2BB'],
            'localauthorityname': ['Council X', 'Council Y'],
            'ratingvalue': ['5', '4'], # Kept as string, type conversion handled by BQ or later if needed by schema
            'newratingpending': [False, True], # Boolean directly for sanitized input
            'first_seen': ['2023-03-01', '2023-03-02'],
            'manual_review': ['reviewed', 'pending'],
            'gemini_insights': [None, 'Insightful text here']
        }
        # Ensure all columns from NEW_BQ_SCHEMA are present
        for col_name in sanitized_schema_names:
            if col_name not in data:
                data[col_name] = [None] * len(data['fhrsid']) # Ensure same length

        df = pd.DataFrame(data)
        # Ensure correct dtypes as per NEW_BQ_SCHEMA expectations for fhrsid (string) and newratingpending (bool)
        # The append_to_bigquery function itself has specific dtype handling for these.
        df['fhrsid'] = df['fhrsid'].astype(str)
        df['newratingpending'] = df['newratingpending'].astype(bool)


        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        # Call append_to_bigquery (no schema argument)
        result = append_to_bigquery(df.copy(), project_id, dataset_id, table_id)

        self.assertTrue(result)
        mock_bq_client.return_value.load_table_from_dataframe.assert_called_once()

        args_list = mock_bq_client.return_value.load_table_from_dataframe.call_args_list
        self.assertEqual(len(args_list), 1)

        called_df = args_list[0][0][0]
        job_config = args_list[0][1]['job_config']

        self.assertEqual(job_config.write_disposition, bigquery.WriteDisposition.WRITE_APPEND)
        self.assertEqual(job_config.schema, NEW_BQ_SCHEMA) # Assert schema is NEW_BQ_SCHEMA

        # Assert types in the DataFrame passed to BQ
        # fhrsid should be string as per NEW_BQ_SCHEMA and internal function logic
        self.assertTrue(pd.api.types.is_string_dtype(called_df[sanitize_column_name('FHRSID')]))
        # newratingpending should be boolean
        self.assertTrue(pd.api.types.is_bool_dtype(called_df[sanitize_column_name('NewRatingPending')]))

        # Check that only columns in NEW_BQ_SCHEMA are present
        self.assertEqual(set(called_df.columns), set(sanitized_schema_names))
        # mock_st.success.assert_called_once() # st is not mocked here

    @patch('builtins.print') # Patched print
    @patch('bq_utils.bigquery.Client')
    def test_append_failure_on_load(self, mock_bq_client, mock_print):
        mock_bq_client.return_value.load_table_from_dataframe.side_effect = Exception("BQ API error")

        # Prepare minimal DataFrame matching NEW_BQ_SCHEMA structure
        sanitized_schema_names = [field.name for field in NEW_BQ_SCHEMA]
        data = {name: [None] for name in sanitized_schema_names} # Minimal data
        data['fhrsid'] = ["1"] # Ensure fhrsid is present and string
        data['newratingpending'] = [False] # Ensure newratingpending is present and bool
        df = pd.DataFrame(data)
        df['fhrsid'] = df['fhrsid'].astype(str)
        df['newratingpending'] = df['newratingpending'].astype(bool)


        result = append_to_bigquery(df.copy(), "p", "d", "t") # No schema argument
        self.assertFalse(result)
        # mock_st.error.assert_called_once() # st is not mocked here

    @patch('builtins.print') # Patched print
    @patch('bq_utils.bigquery.Client')
    def test_append_empty_dataframe(self, mock_bq_client, mock_print):
        sanitized_schema_names = [field.name for field in NEW_BQ_SCHEMA]
        empty_df = pd.DataFrame(columns=sanitized_schema_names)
        # Ensure dtypes for critical columns if df was not empty, important for BQ client
        empty_df['fhrsid'] = empty_df['fhrsid'].astype(str) if 'fhrsid' in empty_df else pd.Series(dtype=str)
        empty_df['newratingpending'] = empty_df['newratingpending'].astype(bool) if 'newratingpending' in empty_df else pd.Series(dtype=bool)


        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        result = append_to_bigquery(empty_df.copy(), project_id, dataset_id, table_id) # No schema

        self.assertTrue(result)
        mock_bq_client.return_value.load_table_from_dataframe.assert_called_once()
        # mock_st.success.assert_called_once() # st is not mocked

    @patch('builtins.print') # Patched print
    @patch('bq_utils.bigquery.Client')
    def test_column_subsetting_and_ordering_for_append(self, mock_bq_client, mock_print):
        mock_job = MagicMock()
        mock_bq_client.return_value.load_table_from_dataframe.return_value = mock_job
        mock_job.result.return_value = None

        # Data with extra field and unsorted columns
        data = {
            'businessname': ['Alice Cafe'],
            'fhrsid': ["100"],
            'extra_field': ['skip_me'],
            'newratingpending': [True],
            # Fill other required fields from NEW_BQ_SCHEMA with None or defaults
        }
        sanitized_schema_names = [field.name for field in NEW_BQ_SCHEMA]
        for col_name in sanitized_schema_names:
            if col_name not in data:
                data[col_name] = [None]

        df = pd.DataFrame(data)
        # Ensure dtypes for specific columns
        df['fhrsid'] = df['fhrsid'].astype(str)
        df['newratingpending'] = df['newratingpending'].astype(bool)


        append_to_bigquery(df.copy(), "p", "d", "t") # No schema argument

        called_df = mock_bq_client.return_value.load_table_from_dataframe.call_args[0][0]

        self.assertNotIn('extra_field', called_df.columns)
        # Assert that columns in called_df are exactly those in NEW_BQ_SCHEMA (sanitized names)
        self.assertEqual(set(called_df.columns), set(sanitized_schema_names))
        # Assert order of columns in called_df matches NEW_BQ_SCHEMA
        self.assertEqual(list(called_df.columns), sanitized_schema_names)


    @patch('builtins.print') # Patched print
    @patch('bq_utils.bigquery.Client')
    def test_append_to_bigquery_fhrsid_string_handling(self, mock_bq_client_constructor, mock_print):
        mock_bq_client_instance = mock_bq_client_constructor.return_value
        mock_load_job = MagicMock()
        mock_bq_client_instance.load_table_from_dataframe.return_value = mock_load_job
        mock_load_job.result.return_value = None

        # Data for fhrsid (should be string as per NEW_BQ_SCHEMA)
        # append_to_bigquery will ensure it's string if schema says string
        sanitized_schema_names = [field.name for field in NEW_BQ_SCHEMA]
        data = {name: [None] for name in sanitized_schema_names} # Initialize
        data.update({
            'fhrsid': [123, "456", 789.0, None], # Mixed types, should be converted to string
            'businessname': ['v1', 'v2', 'v3', 'v4'], # Example other field
            'newratingpending': [False, True, False, True] # ensure boolean
        })
        sample_df = pd.DataFrame(data)
        # Explicitly set other required columns from NEW_BQ_SCHEMA to avoid issues if not in data dict
        for col_name in sanitized_schema_names:
            if col_name not in sample_df.columns:
                 sample_df[col_name] = pd.NA


        result = append_to_bigquery(sample_df.copy(), "p", "d", "t") # No schema argument

        self.assertTrue(result, "append_to_bigquery should return True on success")
        # mock_st.success.assert_called_once() # st not mocked

        mock_bq_client_instance.load_table_from_dataframe.assert_called_once()
        args, kwargs = mock_bq_client_instance.load_table_from_dataframe.call_args
        loaded_df = args[0]

        fhrsid_sanitized = sanitize_column_name('FHRSID')
        self.assertTrue(pd.api.types.is_string_dtype(loaded_df[fhrsid_sanitized]) or loaded_df[fhrsid_sanitized].dtype == 'object',
                        f"{fhrsid_sanitized} column should be string type, but was {loaded_df[fhrsid_sanitized].dtype}")

        # Expected values after .astype(str) conversion by append_to_bigquery's internal logic
        # None becomes 'None', pd.NA becomes '<NA>' (depending on pandas version and specific handling)
        # The function's internal logic does: df_subset[fhrsid_col_name].astype(str)
        expected_str_values = ["123", "456", "789.0", "None"] # How astype(str) handles None
        # Need to adjust if pd.NA is handled differently to become "<NA>" by astype(str)
        # For this test, let's assume None was used for simplicity in input.
        # If pd.NA was in input and converted, it would be "<NA>"

        # Recreate series for comparison based on actual internal astype(str) behavior for various inputs
        input_series_for_expected = pd.Series([123, "456", 789.0, None], dtype=object)
        expected_series_astype_str = input_series_for_expected.astype(str)

        pd.testing.assert_series_equal(loaded_df[fhrsid_sanitized], expected_series_astype_str, check_names=False)

    # Remove tests for fhrsid as integer or coercing to NaN for integer,
    # as NEW_BQ_SCHEMA defines fhrsid as STRING, and append_to_bigquery enforces this.
    # test_append_to_bigquery_fhrsid_is_integer_and_coerces_invalid (remove)
    # test_append_to_bigquery_fhrsid_conversion_error_to_nan_for_int64 (remove)


# If __name__ == '__main__':
#     unittest.main() # This allows running file directly if not using pytest
# For pytest, this is not strictly necessary.
# To run with pytest, ensure pytest and necessary libraries are installed:
# pip install pytest pandas google-cloud-bigquery streamlit
# Then, from the terminal in the directory containing this file:
# pytest test_bq_utils.py
# Or simply: pytest (if it's in a standard test discovery path)
