import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from pandas.testing import assert_series_equal
from google.cloud import bigquery
import sys
import os

# Add the parent directory to the Python path to find st_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from st_app import write_to_bigquery, sanitize_column_name, handle_fetch_data_action

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

    @patch('st_app.st')
    @patch('st_app.write_to_bigquery')
    @patch('st_app.upload_to_gcs')
    @patch('st_app.load_master_data')
    @patch('st_app.fetch_api_data')
    def test_handle_fetch_data_action_rating_date_conversion(
        self, 
        mock_fetch_api_data, 
        mock_load_master_data, 
        mock_upload_to_gcs, 
        mock_write_to_bigquery, 
        mock_st_streamlit_api # Renamed to avoid conflict with mock_st from the class level if it were used
    ):
        # 1. Configure mocks
        mock_fetch_api_data.return_value = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [{
                        'FHRSID': '123',
                        'BusinessName': 'Test Cafe',
                        'RatingDate': '2023-01-16T00:00:00', # String date
                        'PostCode': 'AB1 2CD', # Added to match one of the selected columns
                        'LocalAuthorityName': 'Test Authority', # Added
                        # Add other minimal required fields if json_normalize needs them or if they are part of columns_to_select
                        # For this test, we are mainly concerned with RatingDate.
                        # The columns_to_select in handle_fetch_data_action includes more fields.
                        # To make pd.json_normalize and subsequent selection work without errors,
                        # we should provide those fields or ensure they are handled (e.g. by being optional).
                        'AddressLine1': '1 Test Street',
                        'RatingValue': '5',
                        'Scores.Hygiene': 5, # Assuming this might be expected by schema
                        'Geocode.Longitude': '0.1',
                        'Geocode.Latitude': '51.1',
                        # 'Scores.Structural': None, # Example if some fields can be None
                        # 'Scores.ConfidenceInManagement': None, # Example
                        # 'first_seen': '2023-01-01' # This is added by process_and_update_master_data
                    }]
                }
            }
        }
        mock_load_master_data.return_value = []  # No existing master data
        mock_upload_to_gcs.return_value = True
        # mock_write_to_bigquery is already a mock from the decorator

        # 2. Prepare inputs for handle_fetch_data_action
        coordinate_pairs_str = "0.0,0.0"
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
        mock_write_to_bigquery.assert_called_once()

        # Retrieve the DataFrame passed to write_to_bigquery
        # The arguments are (df, project_id, dataset_id, table_id, columns_to_select, bq_schema)
        args_call_to_bq, _ = mock_write_to_bigquery.call_args
        df_passed_to_bq = args_call_to_bq[0]

        self.assertIsInstance(df_passed_to_bq, pd.DataFrame)
        self.assertTrue('RatingDate' in df_passed_to_bq.columns)
        self.assertEqual(df_passed_to_bq['RatingDate'].dtype, 'datetime64[ns]')
        
        # Check 'first_seen' which is added by process_and_update_master_data
        self.assertTrue('first_seen' in df_passed_to_bq.columns) 
        # The 'first_seen' column is a string 'YYYY-MM-DD', so its dtype should be object or string
        # In the BigQuery schema, it's DATE, but in the DataFrame before BQ client, it's often string/object.
        # The BQ client library handles the conversion if the string is in the correct format.
        self.assertTrue(pd.api.types.is_string_dtype(df_passed_to_bq['first_seen']) or \
                        pd.api.types.is_object_dtype(df_passed_to_bq['first_seen']))

        # Check that streamlit success messages were called (e.g. for API fetch, GCS uploads, BQ write)
        # The exact number of calls can be tricky, so check for specific important ones or >= 1
        mock_st_streamlit_api.success.assert_any_call(f"Total establishments fetched from all API calls: 1")
        mock_st_streamlit_api.success.assert_any_call(f"Processed API response. Added 1 new restaurant records. Total unique records: 1")
        mock_st_streamlit_api.success.assert_any_call(f"Successfully uploaded combined raw API response to {gcs_destination_uri_str}combined_api_response_{pd.Timestamp.now().strftime('%Y-%m-%d')}.json")
        mock_st_streamlit_api.success.assert_any_call(f"Successfully uploaded master restaurant data to {gcs_master_output_uri_str}")
        # The success message for BQ is inside write_to_bigquery, which is mocked. 
        # If we wanted to test st.success from within the *original* write_to_bigquery, we'd need a different approach (e.g. partial mock).
        # But since we mock write_to_bigquery itself, we assert it was called.

        # Check for potential warnings, e.g. if 'RatingDate' was missing (it shouldn't be in this test)
        # Create a list of all calls to st.warning
        warning_calls = [call_args[0][0] for call_args in mock_st_streamlit_api.warning.call_args_list]
        self.assertNotIn("Column 'RatingDate' not found in DataFrame. Skipping datetime conversion for it.", warning_calls)


if __name__ == '__main__':
    unittest.main()
