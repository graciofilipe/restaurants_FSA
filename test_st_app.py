import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from google.cloud import bigquery
import sys
import os

# Add the parent directory to the Python path to find st_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming st_app.py is in the same directory or accessible in PYTHONPATH
# This import will be problematic if st_app.py itself tries to run Streamlit commands
# For testing, we typically mock Streamlit elements if they interfere.
# However, for this specific test, we only need write_to_bigquery and sanitize_column_name.
from st_app import write_to_bigquery, sanitize_column_name

class TestWriteToBigQuery(unittest.TestCase):
    @patch('st_app.bigquery.Client') # Patch where 'bigquery.Client' is looked up (in st_app module)
    @patch('st_app.st') # Mock streamlit module within st_app.py
    def test_write_to_bigquery_with_selection_and_schema(self, mock_st, mock_bigquery_client):
        # Mock the client and its methods
        mock_client_instance = mock_bigquery_client.return_value
        mock_job = MagicMock() # This will be the return value of load_table_from_dataframe
        mock_client_instance.load_table_from_dataframe.return_value = mock_job

        # 1. Create a sample Pandas DataFrame based on the provided fields
        data = {
            'FHRSID': ['101', '102', '103'], # Use strings as per issue, BQ schema will define target type
            'LocalAuthorityBusinessID': ['LA001', 'LA002', 'LA003'],
            'BusinessName': ['Cafe Uno', 'Restaurant Dos', 'Pub Tres'],
            'AddressLine1': ['1 Main St', '2 High St', '3 Park Ave'],
            'AddressLine2': ['Suburb', 'Town', 'City'],
            'AddressLine3': ['', '', 'District'],
            'PostCode': ['SW1A 1AA', 'EC1A 1BB', 'W1A 1CC'],
            'RatingValue': [5, 4, 3], # Integer as per issue
            'RatingKey': ['fhrs_5_en-gb', 'fhrs_4_en-gb', 'fhrs_3_en-gb'],
            'RatingDate': ['2023-01-01', '2023-02-15', '2023-03-20'], # String, BQ schema will define as TIMESTAMP/DATE
            'LocalAuthorityName': ['City Council', 'Borough Council', 'District Council'],
            'NewRatingPending': ['False', 'True', 'False'], # String for boolean representation
            'first_seen': ['2022-12-01', '2023-01-10', '2023-02-25'], # String, BQ schema will define as DATE
            'Scores.Hygiene': [10, 5, 0], # Column name needing sanitization, integer value
            'Scores.Structural': [10, 5, 5],
            'Scores.ConfidenceInManagement': [10, 0, 0],
            'Geocode.Longitude': [-0.1276, 0.0769, -0.1410], # Float
            'Geocode.Latitude': [51.5074, 51.5155, 51.5014],
            'Extra Unselected Column': ['extra_val1', 'extra_val2', 'extra_val3'] # This column should not be selected
        }
        sample_df = pd.DataFrame(data)

        # 2. Define a list of columns_to_select (original names)
        columns_to_select = [
            'FHRSID', 
            'BusinessName', 
            'AddressLine1', 
            'PostCode', 
            'RatingValue', 
            'RatingDate',
            'LocalAuthorityName',
            'Scores.Hygiene', # Original name before sanitization for selection
            'first_seen',
            'Geocode.Longitude',
            'Geocode.Latitude'
        ]

        # 3. Define a bq_schema using SANITIZED names and correct BQ types
        # The names here MUST be what sanitize_column_name(col) would produce for `columns_to_select`
        bq_schema = [
            bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING'), # As per issue list
            bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('AddressLine1'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('PostCode'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('RatingValue'), 'INTEGER'), # As per issue list
            bigquery.SchemaField(sanitize_column_name('RatingDate'), 'STRING'), # As per issue list (could be DATE/TIMESTAMP too)
            bigquery.SchemaField(sanitize_column_name('LocalAuthorityName'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('Scores.Hygiene'), 'INTEGER'), # As per issue list
            bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE'), # As per issue list
            bigquery.SchemaField(sanitize_column_name('Geocode.Longitude'), 'FLOAT'), # As per issue list
            bigquery.SchemaField(sanitize_column_name('Geocode.Latitude'), 'FLOAT')  # As per issue list
        ]
        
        # Expected sanitized column names for the DataFrame that gets loaded into BQ
        expected_sanitized_df_columns = [field.name for field in bq_schema]

        # 4. Call the write_to_bigquery function
        project_id = "test-gcp-project"
        dataset_id = "test_food_dataset"
        table_id = "establishments_table"
        
        # Pass a .copy() of the DataFrame as the function modifies it (sanitizes column names)
        success = write_to_bigquery(
            sample_df.copy(), 
            project_id, 
            dataset_id, 
            table_id, 
            columns_to_select, # Original names for selection
            bq_schema          # Schema with sanitized names
        )

        # 5. Assert that the function reported success and load_table_from_dataframe was called
        self.assertTrue(success)
        mock_client_instance.load_table_from_dataframe.assert_called_once()

        # 6. Capture the arguments passed to load_table_from_dataframe
        call_args = mock_client_instance.load_table_from_dataframe.call_args
        loaded_df_arg = call_args[0][0] # First positional argument to load_table_from_dataframe
        table_ref_str_arg = call_args[0][1] # Second positional argument
        job_config_arg = call_args[1]['job_config'] # Keyword argument

        # Assert table reference string
        self.assertEqual(table_ref_str_arg, f"{project_id}.{dataset_id}.{table_id}")

        # 7. Assert that the DataFrame passed to it:
        #    * Only contains the columns whose original names were in columns_to_select.
        #    * Has column names that have been sanitized.
        self.assertListEqual(list(loaded_df_arg.columns), expected_sanitized_df_columns)
        
        # Verify data integrity for a few columns, ensuring correct mapping after selection and sanitization
        for original_col_name in columns_to_select:
            sanitized_name = sanitize_column_name(original_col_name)
            self.assertTrue(sanitized_name in loaded_df_arg.columns)
            # Using .tolist() for comparison handles potential dtype differences (e.g. int64 vs int)
            pd.testing.assert_series_equal(
                loaded_df_arg[sanitized_name].reset_index(drop=True), 
                sample_df[original_col_name].reset_index(drop=True), 
                check_dtype=False, # BQ loader handles type casting based on schema
                obj=f"DataFrame column '{sanitized_name}'"
            )
        
        # Ensure 'Extra Unselected Column' is not present, even its sanitized version
        self.assertNotIn(sanitize_column_name('Extra Unselected Column'), loaded_df_arg.columns)

        # 8. Assert that the job_config argument has its schema attribute set to bq_schema
        self.assertEqual(job_config_arg.schema, bq_schema)
        
        # Assert other job_config properties
        self.assertEqual(job_config_arg.write_disposition, bigquery.WriteDisposition.WRITE_TRUNCATE)
        self.assertEqual(job_config_arg.column_name_character_map, "V2")
        
        # Assert that the mock job's result method was called (waits for job completion)
        mock_job.result.assert_called_once()
        
        # Assert that st.success and st.error were not called directly in this path
        # (or mock them and check calls if they were part of write_to_bigquery directly)
        # For this test, we rely on the function's return value and BQ client calls.
        # write_to_bigquery uses st.success/st.error, so we should check them if not testing just the core logic.
        # However, the prompt focuses on BQ interaction, so mocking st is fine.
        mock_st.success.assert_called_once() # write_to_bigquery calls st.success on success
        mock_st.error.assert_not_called()


if __name__ == '__main__':
    # This allows running the test directly from the command line
    # Change to the directory containing test_st_app.py and run `python test_st_app.py`
    # Ensure st_app.py is in the parent directory or adjust sys.path accordingly.
    unittest.main()
