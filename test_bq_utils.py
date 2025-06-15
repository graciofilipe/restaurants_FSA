import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, ANY # Added ANY
from bq_utils import (
    BigQueryExecutionError,
    write_to_bigquery, sanitize_column_name, load_all_data_from_bq,
    append_to_bigquery, ORIGINAL_COLUMNS_TO_KEEP
)
from google.cloud import bigquery, exceptions # Import exceptions for error testing
from google.auth.exceptions import DefaultCredentialsError # Added import

# Attempt to import GenericGBQException for more specific error testing if available
try:
    from pandas_gbq.gbq import GenericGBQException
except ImportError:
    GenericGBQException = None # Fallback if pandas_gbq is not installed or structure differs

# Define NEW_BQ_SCHEMA directly in the test file
NEW_BQ_SCHEMA = [
    bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('AddressLine1'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('AddressLine2'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('AddressLine3'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('PostCode'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('LocalAuthorityName'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('RatingValue'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('NewRatingPending'), 'BOOLEAN', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('manual_review'), 'STRING', mode='NULLABLE'),
    bigquery.SchemaField(sanitize_column_name('gemini_insights'), 'STRING', mode='NULLABLE'),
]

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
        table_id=table_id,
        columns_to_select=ORIGINAL_COLUMNS_TO_KEEP, # Added columns_to_select
        bq_schema=NEW_BQ_SCHEMA # Added bq_schema
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
        result = append_to_bigquery(df.copy(), project_id, dataset_id, table_id, bq_schema=NEW_BQ_SCHEMA)

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


        result = append_to_bigquery(df.copy(), "p", "d", "t", bq_schema=NEW_BQ_SCHEMA)
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

        result = append_to_bigquery(empty_df.copy(), project_id, dataset_id, table_id, bq_schema=NEW_BQ_SCHEMA)

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


        append_to_bigquery(df.copy(), "p", "d", "t", bq_schema=NEW_BQ_SCHEMA)

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
        num_records = 4 # Define the number of records for consistent array lengths

        # Initialize data with arrays of consistent length
        data = {name: [None] * num_records for name in sanitized_schema_names}

        # Update specific columns with test data
        data.update({
            'fhrsid': [123, "456", 789.0, None],
            'businessname': ['v1', 'v2', 'v3', 'v4'],
            'newratingpending': [False, True, False, True]
        })
        sample_df = pd.DataFrame(data)

        # Ensure all columns from NEW_BQ_SCHEMA are present; this step might be redundant now
        # but kept for safety, ensuring all schema columns are in the DataFrame.
        for col_name in sanitized_schema_names:
            if col_name not in sample_df.columns:
                 sample_df[col_name] = pd.Series([pd.NA] * num_records, dtype=object)


        result = append_to_bigquery(sample_df.copy(), "p", "d", "t", bq_schema=NEW_BQ_SCHEMA)

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
        # Ensure all elements in the expected series are explicitly strings, including 'None' for None.
        expected_list = ["123", "456", "789.0", "None"]

        # Convert the actual series to a list of Python strings for comparison
        actual_list = loaded_df[fhrsid_sanitized].tolist()

        self.assertEqual(actual_list, expected_list)

    # Remove tests for fhrsid as integer or coercing to NaN for integer,
    # as NEW_BQ_SCHEMA defines fhrsid as STRING, and append_to_bigquery enforces this.
    # test_append_to_bigquery_fhrsid_is_integer_and_coerces_invalid (remove)
    # test_append_to_bigquery_fhrsid_conversion_error_to_nan_for_int64 (remove)

# --- Tests for update_rows_in_bigquery ---
import unittest # Already imported above, but good for clarity if this section moved
from unittest.mock import patch, MagicMock # Already imported
from google.cloud import bigquery, exceptions as google_exceptions # Ensure exceptions is imported
from bq_utils import update_rows_in_bigquery, FHRSID_COLNAME

class TestUpdateRowsInBigQuery(unittest.TestCase):
    @patch('bq_utils.bigquery.Client')
    def test_successful_update(self, mock_bq_client_constructor):
        mock_client_instance = mock_bq_client_constructor.return_value
        mock_query_job = MagicMock()
        mock_client_instance.query.return_value = mock_query_job
        mock_query_job.result.return_value = None # Simulate job completion
        mock_query_job.errors = None # Simulate no errors

        project_id = 'p'
        dataset_id = 'd'
        table_id = 't'
        fhrsid = '123'
        update_data = {
            'colA': 'new_val',
            'colB': 100,
            'colC': True,
            'colD': None,
            "colE_quote": "val'ue"
        }

        result = update_rows_in_bigquery(project_id, dataset_id, table_id, fhrsid, update_data)

        self.assertTrue(result)
        mock_client_instance.query.assert_called_once()

        actual_query = mock_client_instance.query.call_args[0][0]
        expected_query_set_clause = "SET `colA` = 'new_val', `colB` = 100, `colC` = TRUE, `colD` = NULL, `colE_quote` = 'val''ue'"
        expected_query_where_clause = f"WHERE {FHRSID_COLNAME} = '123'"

        self.assertIn(expected_query_set_clause, actual_query)
        self.assertIn(expected_query_where_clause, actual_query)
        # Check table name
        self.assertIn(f"UPDATE `{project_id}.{dataset_id}.{table_id}`", actual_query)

    @patch('bq_utils.bigquery.Client')
    def test_fhrsid_with_single_quote(self, mock_bq_client_constructor):
        mock_client_instance = mock_bq_client_constructor.return_value
        mock_query_job = MagicMock()
        mock_client_instance.query.return_value = mock_query_job
        mock_query_job.result.return_value = None
        mock_query_job.errors = None

        project_id = 'p'
        dataset_id = 'd'
        table_id = 't'
        fhrsid_with_quote = "test'fhrsid"
        update_data = {'colA': 'new_val'}

        result = update_rows_in_bigquery(project_id, dataset_id, table_id, fhrsid_with_quote, update_data)

        self.assertTrue(result)
        actual_query = mock_client_instance.query.call_args[0][0]
        expected_where_clause = f"WHERE {FHRSID_COLNAME} = 'test''fhrsid'"
        self.assertIn(expected_where_clause, actual_query)

    @patch('bq_utils.bigquery.Client')
    def test_bigquery_api_error(self, mock_bq_client_constructor):
        mock_client_instance = mock_bq_client_constructor.return_value
        # Simulate an error during query execution
        mock_client_instance.query.side_effect = google_exceptions.GoogleCloudError("Simulated API error")

        project_id = 'p'
        dataset_id = 'd'
        table_id = 't'
        fhrsid = '123'
        update_data = {'colA': 'new_val'}

        result = update_rows_in_bigquery(project_id, dataset_id, table_id, fhrsid, update_data)
        self.assertFalse(result)

    @patch('bq_utils.bigquery.Client')
    def test_bigquery_job_error(self, mock_bq_client_constructor):
        mock_client_instance = mock_bq_client_constructor.return_value
        mock_query_job = MagicMock()
        mock_client_instance.query.return_value = mock_query_job
        mock_query_job.result.return_value = None # Simulate job completion
        mock_query_job.errors = [{'message': 'Job failed'}] # Simulate errors in the job

        project_id = 'p'
        dataset_id = 'd'
        table_id = 't'
        fhrsid = '123'
        update_data = {'colA': 'new_val'}

        result = update_rows_in_bigquery(project_id, dataset_id, table_id, fhrsid, update_data)
        self.assertFalse(result)


    @patch('bq_utils.bigquery.Client')
    def test_empty_update_data(self, mock_bq_client_constructor):
        mock_client_instance = mock_bq_client_constructor.return_value

        project_id = 'p'
        dataset_id = 'd'
        table_id = 't'
        fhrsid = '123'
        update_data = {} # Empty update data

        # The function `update_rows_in_bigquery` itself returns False if update_data is empty
        # before attempting any BQ calls.
        result = update_rows_in_bigquery(project_id, dataset_id, table_id, fhrsid, update_data)

        self.assertFalse(result)
        mock_client_instance.query.assert_not_called() # Ensure BQ query is not made


# If __name__ == '__main__':
#     unittest.main() # This allows running file directly if not using pytest
# For pytest, this is not strictly necessary.
# To run with pytest, ensure pytest and necessary libraries are installed:
# pip install pytest pandas google-cloud-bigquery streamlit
# Then, from the terminal in the directory containing this file:
# pytest test_bq_utils.py
# Or simply: pytest (if it's in a standard test discovery path)
