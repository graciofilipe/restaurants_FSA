import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from pandas.testing import assert_series_equal
from google.cloud import bigquery
import sys
import os

# Add the parent directory to the Python path to find st_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Updated imports based on refactoring
from bq_utils import write_to_bigquery, sanitize_column_name
from st_app import handle_fetch_data_action
# google.cloud.bigquery is imported directly where needed (e.g., TestWriteToBigQuery uses bigquery.SchemaField)

class TestWriteToBigQuery(unittest.TestCase):
    @patch('bq_utils.bigquery.Client') # Patched in bq_utils
    @patch('bq_utils.st') # Patched in bq_utils
    def test_write_to_bigquery_with_selection_and_schema(self, mock_st, mock_bigquery_client):
        # Mock the client and its methods
        mock_client_instance = mock_bigquery_client.return_value
        mock_job = MagicMock() # This will be the return value of load_table_from_dataframe
        mock_client_instance.load_table_from_dataframe.return_value = mock_job

        # 1. Create a sample Pandas DataFrame based on the provided fields
        data = {
            'FHRSID': ['101', '102', '103'], 
            'LocalAuthorityBusinessID': ['LA001', 'LA002', 'LA003'],
            'BusinessName': ['Cafe Uno', 'Restaurant Dos', 'Pub Tres'],
            'AddressLine1': ['1 Main St', '2 High St', '3 Park Ave'],
            'AddressLine2': ['Suburb', 'Town', 'City'],
            'AddressLine3': ['', '', 'District'],
            'PostCode': ['SW1A 1AA', 'EC1A 1BB', 'W1A 1CC'],
            'RatingValue': [5, 4, 3], 
            'RatingKey': ['fhrs_5_en-gb', 'fhrs_4_en-gb', 'fhrs_3_en-gb'],
            'RatingDate': ['2023-01-01', '2023-02-15', '2023-03-20'], 
            'LocalAuthorityName': ['City Council', 'Borough Council', 'District Council'],
            'NewRatingPending': ['False', 'True', 'False'], 
            'first_seen': ['2022-12-01', '2023-01-10', '2023-02-25'], 
            'Scores.Hygiene': [10, 5, 0], 
            'Scores.Structural': [10, 5, 5],
            'Scores.ConfidenceInManagement': [10, 0, 0],
            'Geocode.Longitude': [-0.1276, 0.0769, -0.1410], 
            'Geocode.Latitude': [51.5074, 51.5155, 51.5014],
            'Extra Unselected Column': ['extra_val1', 'extra_val2', 'extra_val3'] 
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
            'Scores.Hygiene', 
            'first_seen',
            'Geocode.Longitude',
            'Geocode.Latitude'
        ]

        # 3. Define a bq_schema using SANITIZED names and correct BQ types
        bq_schema = [
            bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING'), 
            bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('AddressLine1'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('PostCode'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('RatingValue'), 'INTEGER'), 
            bigquery.SchemaField(sanitize_column_name('RatingDate'), 'STRING'), 
            bigquery.SchemaField(sanitize_column_name('LocalAuthorityName'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('Scores.Hygiene'), 'INTEGER'), 
            bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE'), 
            bigquery.SchemaField(sanitize_column_name('Geocode.Longitude'), 'FLOAT'), 
            bigquery.SchemaField(sanitize_column_name('Geocode.Latitude'), 'FLOAT')
        ]
        
        expected_sanitized_df_columns = [field.name for field in bq_schema]

        # 4. Call the write_to_bigquery function
        project_id = "test-gcp-project"
        dataset_id = "test_food_dataset"
        table_id = "establishments_table"
        
        success = write_to_bigquery(
            sample_df.copy(), 
            project_id, 
            dataset_id, 
            table_id, 
            columns_to_select, 
            bq_schema
        )

        self.assertTrue(success)
        mock_client_instance.load_table_from_dataframe.assert_called_once()

        call_args = mock_client_instance.load_table_from_dataframe.call_args
        loaded_df_arg = call_args[0][0] 
        table_ref_str_arg = call_args[0][1] 
        job_config_arg = call_args[1]['job_config'] 

        self.assertEqual(table_ref_str_arg, f"{project_id}.{dataset_id}.{table_id}")
        self.assertListEqual(list(loaded_df_arg.columns), expected_sanitized_df_columns)
        
        for original_col_name in columns_to_select:
            sanitized_name = sanitize_column_name(original_col_name)
            self.assertTrue(sanitized_name in loaded_df_arg.columns)
            assert_series_equal(
                loaded_df_arg[sanitized_name].reset_index(drop=True), 
                sample_df[original_col_name].reset_index(drop=True), 
                check_dtype=False,
                check_names=False, # Add this to ignore series name differences
                obj=f"DataFrame column '{sanitized_name}'"
            )
        
        self.assertNotIn(sanitize_column_name('Extra Unselected Column'), loaded_df_arg.columns)
        self.assertEqual(job_config_arg.schema, bq_schema)
        self.assertEqual(job_config_arg.write_disposition, bigquery.WriteDisposition.WRITE_TRUNCATE)
        self.assertEqual(job_config_arg.column_name_character_map, "V2")
        mock_job.result.assert_called_once()
        mock_st.success.assert_called_once() 
        mock_st.error.assert_not_called()

    # Patches updated to reflect new module locations for the mocked functions
    @patch('st_app.st') # For st calls directly within handle_fetch_data_action
    @patch('st_app.time') # For time.sleep directly within handle_fetch_data_action
    @patch('bq_utils.write_to_bigquery') # Mocked function now in bq_utils
    @patch('gcs_utils.upload_to_gcs') # Mocked function now in gcs_utils
    @patch('data_processing.load_master_data') # Mocked function now in data_processing
    @patch('api_client.fetch_api_data') # Mocked function now in api_client
    def test_handle_fetch_data_action_rating_date_conversion(
        self,
        mock_fetch_api_data,
        mock_load_master_data,
        mock_upload_to_gcs,
        mock_write_to_bigquery,
        mock_st, # Renamed to mock_st for clarity, matches the decorator name
        mock_time
    ):
        # 1. Configure mocks
        # Define two different API responses for two calls
        api_response_1 = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [{
                        'FHRSID': '123', 'BusinessName': 'Test Cafe 1', 
                        'RatingDate': '2023-01-16T00:00:00', 'PostCode': 'AB1 2CD',
                        'LocalAuthorityName': 'Test Authority 1', 'AddressLine1': '1 Test Street',
                        'RatingValue': '5', 'Scores.Hygiene': 5, 
                        'Geocode.Longitude': '0.1', 'Geocode.Latitude': '51.1'
                    }]
                }
            }
        }
        api_response_2 = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [{
                        'FHRSID': '456', 'BusinessName': 'Test Cafe 2',
                        'RatingDate': '2023-02-20T00:00:00', 'PostCode': 'EF3 4GH',
                        'LocalAuthorityName': 'Test Authority 2', 'AddressLine1': '2 Other Street',
                        'RatingValue': '4', 'Scores.Hygiene': 4,
                        'Geocode.Longitude': '0.2', 'Geocode.Latitude': '51.2'
                    }]
                }
            }
        }
        mock_fetch_api_data.side_effect = [api_response_1, api_response_2]
        mock_load_master_data.return_value = []  # No existing master data
        mock_upload_to_gcs.return_value = True
        # mock_write_to_bigquery is already a mock from the decorator

        # 2. Prepare inputs for handle_fetch_data_action
        coordinate_pairs_str = "0.0,0.0\n1.0,1.0" # Two coordinate pairs
        max_results = 10
        gcs_destination_uri_str = "gs://bucket/folder/"
        master_list_uri_str = "gs://bucket/master.json" # Will be loaded by mock_load_master_data
        gcs_master_output_uri_str = "gs://bucket/master_out.json" # Will be used by mock_upload_to_gcs
        bq_full_path_str = "project.dataset.table" # Enables the BQ write path

        # 3. Call handle_fetch_data_action
        # This function is imported from st_app
        handle_fetch_data_action(
            coordinate_pairs_str,
            max_results,
            gcs_destination_uri_str,
            master_list_uri_str,
            gcs_master_output_uri_str,
            bq_full_path_str
        )

        # 4. Assertions
        self.assertEqual(mock_fetch_api_data.call_count, 2) # Called for each coordinate pair
        mock_time.sleep.assert_has_calls([call(4), call(4)]) # Called with 4 seconds
        self.assertEqual(mock_time.sleep.call_count, 2) # Called after each API fetch

        mock_write_to_bigquery.assert_called_once()

        # Retrieve the DataFrame passed to write_to_bigquery
        args_call_to_bq, kwargs_call_to_bq = mock_write_to_bigquery.call_args
        df_passed_to_bq = args_call_to_bq[0]
        # project_id is args_call_to_bq[1]
        # dataset_id is args_call_to_bq[2]
        # table_id is args_call_to_bq[3]
        columns_selected_for_bq = args_call_to_bq[4]
        schema_for_bq = args_call_to_bq[5]


        self.assertIsInstance(df_passed_to_bq, pd.DataFrame)
        self.assertEqual(len(df_passed_to_bq), 2) # Should contain two records
        self.assertTrue('RatingDate' in df_passed_to_bq.columns)
        self.assertTrue(pd.api.types.is_string_dtype(df_passed_to_bq['RatingDate']) or pd.api.types.is_object_dtype(df_passed_to_bq['RatingDate']), "RatingDate column should have string or object dtype")
        
        self.assertTrue('first_seen' in df_passed_to_bq.columns) 
        self.assertTrue(pd.api.types.is_string_dtype(df_passed_to_bq['first_seen']) or \
                        pd.api.types.is_object_dtype(df_passed_to_bq['first_seen']))

        # Assertions for "manual_review"
        self.assertIn("manual_review", columns_selected_for_bq)

        expected_manual_review_schema_field = bigquery.SchemaField("manual_review", "STRING", mode="NULLABLE")

        # Check if a field with the name "manual_review" exists and matches the expected definition
        found_manual_review_field = False
        for field in schema_for_bq:
            if field.name == "manual_review":
                found_manual_review_field = True
                self.assertEqual(field.field_type, expected_manual_review_schema_field.field_type)
                self.assertEqual(field.mode, expected_manual_review_schema_field.mode)
                break
        self.assertTrue(found_manual_review_field, "SchemaField for 'manual_review' not found in bq_schema")

        # Check Streamlit success messages
        # Note: The exact date in the filename might be tricky if the test runs near midnight.
        # Consider mocking datetime.now() if this becomes an issue. For now, assume it's stable enough.
        current_date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Check for specific success calls using assert_any_call
        # We expect 2 total from API calls, and since they are unique, 2 new records added.
        mock_st.success.assert_any_call("Total establishments fetched from all API calls: 2")
        mock_st.success.assert_any_call("Processed API response. Added 2 new restaurant records. Total unique records: 2")
        mock_st.success.assert_any_call(f"Successfully uploaded combined raw API response to {gcs_destination_uri_str}combined_api_response_{current_date_str}.json")
        mock_st.success.assert_any_call(f"Successfully uploaded master restaurant data to {gcs_master_output_uri_str}")
        
        # Check for potential warnings
        warning_calls = [call_args[0][0] for call_args in mock_st.warning.call_args_list]
        self.assertNotIn("Column 'RatingDate' not found in DataFrame. Skipping datetime conversion for it.", warning_calls)

    # This test does not need modification as it tests pandas functionality directly
    # and does not involve the refactored app structure in terms of imports or mocks.
    def test_first_seen_conversion_to_datetime(self):
        """
        Tests that the 'first_seen' column is correctly converted to datetime64[ns]
        and that unparseable dates are handled by becoming NaT.
        This test simulates the conversion that happens in handle_fetch_data_action
        right before data is passed to write_to_bigquery.
        """
        # 1. Create a sample Pandas DataFrame
        data = {
            'FHRSID': [1, 2, 3, 4],
            'first_seen': ["2023-01-15", "2024-02-20", "not-a-date", None],
            'other_col': ['a', 'b', 'c', 'd']
        }
        df = pd.DataFrame(data)

        # 2. Apply the datetime conversion logic (as done in handle_fetch_data_action)
        if 'first_seen' in df.columns:
            df['first_seen'] = pd.to_datetime(df['first_seen'], errors='coerce')

        # 3. Assert that the dtype of the 'first_seen' column is datetime64[ns]
        self.assertEqual(df['first_seen'].dtype, 'datetime64[ns]')

        # 4. Assert that values that were unparseable are now pd.NaT
        #    and valid dates are correctly converted.
        self.assertEqual(df['first_seen'].iloc[0], pd.Timestamp("2023-01-15"))
        self.assertEqual(df['first_seen'].iloc[1], pd.Timestamp("2024-02-20"))
        self.assertTrue(pd.isna(df['first_seen'].iloc[2])) # Check for NaT for "not-a-date"
        self.assertTrue(pd.isna(df['first_seen'].iloc[3])) # Check for NaT for None


if __name__ == '__main__':
    unittest.main()
