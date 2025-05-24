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
    @patch('st_app.bigquery.Client')
    @patch('st_app.st.warning') # Patch st.warning
    @patch('st_app.st.success') # Patch st.success (already part of mock_st if using @patch('st_app.st'))
    @patch('st_app.st.error')   # Patch st.error
    def test_write_to_bigquery_with_selection_and_schema(self, mock_st_error, mock_st_success, mock_st_warning, mock_bigquery_client):
        # Mock the client and its methods
        mock_client_instance = mock_bigquery_client.return_value
        mock_job = MagicMock()
        mock_client_instance.load_table_from_dataframe.return_value = mock_job

        # 1. Create a sample Pandas DataFrame
        data = {
            'FHRSID': ['101', '102', '103', '104', '105', '106'], # STRING type
            'LocalAuthorityBusinessID': ['LA001', 'LA002', 'LA003', 'LA004', 'LA005', 'LA006'],
            'BusinessName': ['Cafe Uno', 'Rest Dos', 'Pub Tres', 'Bistro Q', 'Grill C', 'Sushi S'],
            'AddressLine1': ['1 Main St', '2 High St', '3 Park Ave', '4 Elm St', '5 Oak Ln', '6 Pine Rd'],
            'AddressLine2': ['Suburb', 'Town', 'City', 'Village', 'Hamlet', 'Brow'],
            'AddressLine3': ['', '', 'District', '', 'County', ''],
            'PostCode': ['SW1A 1AA', 'EC1A 1BB', 'W1A 1CC', 'N1 1DD', 'E1 1EE', 'S1 1FF'],
            'RatingValue': ["5", 3, "Exempt", "4.0", "", None], # Mixed types for coercion test
            'RatingKey': ['fhrs_5_en-gb', 'fhrs_3_en-gb', 'fhrs_exempt_en-gb', 'fhrs_4_en-gb', 'fhrs_awaitinginspection_en-gb', 'fhrs_awaitingpublication_en-gb'],
            'RatingDate': ['2023-01-01', '2023-02-15', '2023-03-20', '2023-04-10', '2023-05-01', '2023-06-01'],
            'LocalAuthorityName': ['Council A', 'Council B', 'Council C', 'Council D', 'Council E', 'Council F'],
            'NewRatingPending': ['False', 'False', 'True', 'False', 'True', 'False'],
            'first_seen': ['2022-12-01', '2023-01-10', '2023-02-25', '2023-03-15', '2023-04-05', '2023-05-20'],
            'Scores.Hygiene': [10, 5, None, 0, 15, 5],
            'Scores.Structural': [10, 5, 5, 0, 10, 5],
            'Scores.ConfidenceInManagement': [10, 0, 0, 5, 10, 0],
            'geocode.longitude': [-0.1, 0.07, -0.14, 0.01, -0.05, 0.12],
            'geocode.latitude': [51.5, 51.51, 51.50, 51.52, 51.49, 51.53], # Will be geocode_latitude
            'geocode latitude': [52.0, 53.0, 54.0, 55.0, 56.0, 57.0], # Will be geocode_latitude_1
            'Another Column with Spaces': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
            'Extra Unselected Column': ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6'],
            'col-with-dashes': [1,2,3,4,5,6]
        }
        sample_df = pd.DataFrame(data)

        # 2. Define a list of columns_to_select (original names)
        # This tests selection and includes columns that will test de-duplication
        columns_to_select = [
            'FHRSID', 
            'BusinessName', 
            'RatingValue', 
            'RatingDate',
            'Scores.Hygiene',
            'geocode.longitude',
            'geocode.latitude', 
            'geocode latitude', 
            'Another Column with Spaces',
            'col-with-dashes',
            'NewRatingPending'
        ]

        # 3. Define a bq_schema using FINAL SANITIZED and DE-DUPLICATED names and correct BQ types.
        bq_schema = [
            bigquery.SchemaField('fhrsid', 'STRING'),  # Updated to STRING
            bigquery.SchemaField('businessname', 'STRING'),
            bigquery.SchemaField('ratingvalue', 'INTEGER'), # Updated to INTEGER
            bigquery.SchemaField('ratingdate', 'STRING'),
            bigquery.SchemaField('scores_hygiene', 'INTEGER'),
            bigquery.SchemaField('geocode_longitude', 'FLOAT'),
            bigquery.SchemaField('geocode_latitude', 'FLOAT'),
            bigquery.SchemaField('geocode_latitude_1', 'FLOAT'),
            bigquery.SchemaField('another_column_with_spaces', 'STRING'),
            bigquery.SchemaField('col_with_dashes', 'INTEGER'), 
            bigquery.SchemaField('newratingpending', 'STRING')
        ]
        
        # Expected sanitized column names for the DataFrame that gets loaded into BQ
        # These are the names from the bq_schema, which should match the loaded_df_arg columns.
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
            original_to_final_sanitized_map = {
                'FHRSID': 'fhrsid',
                'BusinessName': 'businessname',
                'RatingValue': 'ratingvalue', # This is the base sanitized name. De-duplication not expected for this.
                'RatingDate': 'ratingdate',
                'Scores.Hygiene': 'scores_hygiene',
                'geocode.longitude': 'geocode_longitude',
                'geocode.latitude': 'geocode_latitude',
                'geocode latitude': 'geocode_latitude_1',
                'Another Column with Spaces': 'another_column_with_spaces',
                'col-with-dashes': 'col_with_dashes',
                'NewRatingPending': 'newratingpending'
            }
            self.assertTrue(original_col_name in original_to_final_sanitized_map, f"Test error: Original col '{original_col_name}' not in map.")
            expected_final_sanitized_name = original_to_final_sanitized_map[original_col_name]
            self.assertTrue(expected_final_sanitized_name in loaded_df_arg.columns, f"Sanitized col '{expected_final_sanitized_name}' not in loaded DF.")

            expected_series_data = sample_df[original_col_name]
            actual_series_data = loaded_df_arg[expected_final_sanitized_name]

            if original_col_name == 'FHRSID':
                # FHRSID is now STRING, ensure it's loaded as such and values match
                pd.testing.assert_series_equal(
                    actual_series_data.reset_index(drop=True),
                    expected_series_data.astype(str).reset_index(drop=True), # Compare as string
                    check_dtype=False, # Loaded series might be object, expected is string
                    obj=f"DataFrame column '{expected_final_sanitized_name}' (original: '{original_col_name}')"
                )
            elif original_col_name == 'RatingValue':
                # Test RatingValue coercion
                # Expected: ["5", 3, "Exempt", "4.0", "", None] -> [5, 3, pd.NA, 4, pd.NA, pd.NA]
                expected_coerced_values = [5, 3, pd.NA, 4, pd.NA, pd.NA]
                for i, val in enumerate(expected_coerced_values):
                    if pd.isna(val):
                        self.assertTrue(pd.isna(actual_series_data.iloc[i]), f"RatingValue at index {i} should be NA, was {actual_series_data.iloc[i]}")
                    else:
                        self.assertEqual(actual_series_data.iloc[i], val, f"RatingValue at index {i} incorrect.")
                self.assertTrue(isinstance(actual_series_data.dtype, pd.Int64Dtype), "RatingValue column is not Int64Dtype")
            elif pd.api.types.is_numeric_dtype(expected_series_data) and not isinstance(actual_series_data.dtype, pd.Int64Dtype): # Avoid Int64Dtype for general numeric
                 pd.testing.assert_series_equal(
                    actual_series_data.reset_index(drop=True), 
                    expected_series_data.reset_index(drop=True),
                    check_dtype=False, 
                    obj=f"DataFrame column '{expected_final_sanitized_name}' (original: '{original_col_name}')",
                    check_exact=False, rtol=1e-5)
            else: # For string types or already correctly typed numerics (like Int64 for Scores.Hygiene if it had NAs)
                 pd.testing.assert_series_equal(
                    actual_series_data.reset_index(drop=True), 
                    expected_series_data.reset_index(drop=True),
                    check_dtype=False, 
                    obj=f"DataFrame column '{expected_final_sanitized_name}' (original: '{original_col_name}')")

        # Ensure unselected columns are not present
        self.assertNotIn('Extra Unselected Column', loaded_df_arg.columns)
        self.assertNotIn(sanitize_column_name('Extra Unselected Column'), loaded_df_arg.columns)
        self.assertNotIn('AddressLine2', loaded_df_arg.columns)
        self.assertNotIn(sanitize_column_name('AddressLine2'), loaded_df_arg.columns)
        self.assertNotIn('LocalAuthorityBusinessID', loaded_df_arg.columns)
        self.assertNotIn(sanitize_column_name('LocalAuthorityBusinessID'), loaded_df_arg.columns)

        # 8. Assert job_config schema and other properties
        self.assertEqual(job_config_arg.schema, bq_schema, "JobConfig schema mismatch.")
        self.assertEqual(job_config_arg.write_disposition, bigquery.WriteDisposition.WRITE_TRUNCATE)
        self.assertEqual(job_config_arg.column_name_character_map, "V2")
        
        mock_job.result.assert_called_once()
        
        # Assert st.warning call due to "Exempt" and "" in RatingValue being coerced
        # Original data: ["5", 3, "Exempt", "4.0", "", None]
        # "Exempt" is 1, "" is 1. So 2 values coerced.
        mock_st_warning.assert_called_once()
        # Check if the warning message contains the count of coerced values.
        # The exact message might be brittle to test, so checking for key parts.
        args, _ = mock_st_warning.call_args
        self.assertIn("2 non-numeric value(s) found in column 'RatingValue'", args[0])
        self.assertIn("coerced to NULL", args[0])

        mock_st_success.assert_called_once() 
        mock_st_error.assert_not_called()

if __name__ == '__main__':
    # This allows running the test directly from the command line
    # Change to the directory containing test_st_app.py and run `python test_st_app.py`
    # Ensure st_app.py is in the parent directory or adjust sys.path accordingly.
    unittest.main()
