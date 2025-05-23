import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
from datetime import datetime as dt # Alias to avoid conflict with datetime module in st_app
import pandas as pd

import requests # For requests.exceptions.RequestException

import requests # For requests.exceptions.RequestException

# Import the Streamlit app module to be tested
import st_app
from st_app import (
    _parse_gcs_uri, upload_to_gcs, fetch_api_data,
    load_master_data, load_json_from_uri, process_and_update_master_data,
    write_to_bigquery # Added import for write_to_bigquery
)
from google.cloud import bigquery # For WriteDisposition


# Mock Streamlit globally for all tests as it's not running in a true Streamlit environment
# This helps avoid errors when st_app.py is imported and tries to use st.* functions immediately
# (though st_app.py as provided doesn't do this at the top level)
st_app.st = MagicMock()

# Mock google.cloud.storage and exceptions for tests
# This allows us to patch storage.Client within st_app
try:
    from google.cloud import storage, exceptions
    st_app.storage = storage
    st_app.exceptions = exceptions # Make it available if st_app needs it for specific exception handling
except ImportError:
    # If google-cloud-storage is not installed, create mock objects
    # This is important for the tests to run in environments where GCS client is not installed
    st_app.storage = MagicMock()
    st_app.exceptions = MagicMock()
    st_app.exceptions.NotFound = type('NotFound', (Exception,), {})


class TestLoadJsonFromUri(unittest.TestCase):

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    def test_successful_load_gcs(self, mock_gcs_client, mock_st_error):
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_string.return_value = json.dumps({"key": "value"})
        
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob
        
        mock_gcs_client_instance = MagicMock()
        mock_gcs_client_instance.bucket.return_value = mock_bucket_instance
        mock_gcs_client.return_value = mock_gcs_client_instance

        uri = "gs://test-bucket/test-file.json"
        result = st_app.load_json_from_uri(uri)

        self.assertEqual(result, {"key": "value"})
        mock_st_error.assert_not_called()
        mock_gcs_client.assert_called_once()
        mock_gcs_client_instance.bucket.assert_called_once_with("test-bucket")
        mock_bucket_instance.blob.assert_called_once_with("test-file.json")
        mock_blob.exists.assert_called_once()
        mock_blob.download_as_string.assert_called_once()

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    def test_gcs_blob_not_found(self, mock_gcs_client, mock_st_error):
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False # Simulate blob not existing
        
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob
        
        mock_gcs_client_instance = MagicMock()
        mock_gcs_client_instance.bucket.return_value = mock_bucket_instance
        mock_gcs_client.return_value = mock_gcs_client_instance

        uri = "gs://test-bucket/nonexistent-file.json"
        result = st_app.load_json_from_uri(uri)

        self.assertIsNone(result)
        mock_st_error.assert_called_once_with(f"Error: GCS file not found at {uri}")

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    def test_gcs_invalid_json(self, mock_gcs_client, mock_st_error):
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_string.return_value = "this is not json" # Invalid JSON
        
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob
        
        mock_gcs_client_instance = MagicMock()
        mock_gcs_client_instance.bucket.return_value = mock_bucket_instance
        mock_gcs_client.return_value = mock_gcs_client_instance
        
        uri = "gs://test-bucket/invalid.json"
        result = st_app.load_json_from_uri(uri)

        self.assertIsNone(result)
        mock_st_error.assert_called_once()
        # Check that the error message contains "Error decoding JSON"
        self.assertIn("Error decoding JSON", mock_st_error.call_args[0][0])

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    def test_gcs_client_exception(self, mock_gcs_client, mock_st_error):
        mock_gcs_client.side_effect = Exception("GCS Client Error") # Simulate client init error
        
        uri = "gs://test-bucket/anyfile.json"
        result = st_app.load_json_from_uri(uri)

        self.assertIsNone(result)
        mock_st_error.assert_called_once_with(f"Error accessing GCS file {uri}: GCS Client Error")

    @patch('st_app.st.error')
    @patch('st_app.storage.Client') 
    def test_gcs_no_blob_name(self, mock_gcs_client, mock_st_error):
        uri = "gs://test-bucket/" # URI pointing to bucket only
        result = st_app.load_json_from_uri(uri)
        self.assertIsNone(result)
        # _parse_gcs_uri returns None, load_json_from_uri then calls st.error
        mock_st_error.assert_called_once_with(f"Invalid GCS URI format: {uri}")
        mock_gcs_client.assert_not_called() # Client should not be initialized if parsing fails

    @patch('st_app.st.error')
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({"key": "value"}))
    def test_successful_load_local(self, mock_file_open, mock_st_error):
        uri = "/path/to/local/file.json"
        result = st_app.load_json_from_uri(uri)
        
        self.assertEqual(result, {"key": "value"})
        mock_file_open.assert_called_once_with(uri, 'r')
        mock_st_error.assert_not_called()

    @patch('st_app.st.error')
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_local_file_not_found(self, mock_file_open, mock_st_error):
        uri = "/path/to/nonexistent/file.json"
        result = st_app.load_json_from_uri(uri)
        
        self.assertIsNone(result)
        mock_file_open.assert_called_once_with(uri, 'r')
        mock_st_error.assert_called_once_with(f"Error: Local file not found at {uri}")

    @patch('st_app.st.error')
    @patch('builtins.open', new_callable=mock_open, read_data="this is not json")
    def test_local_invalid_json(self, mock_file_open, mock_st_error):
        uri = "/path/to/invalidlocal.json"
        result = st_app.load_json_from_uri(uri)
        
        self.assertIsNone(result)
        mock_file_open.assert_called_once_with(uri, 'r')
        mock_st_error.assert_called_once()
        self.assertIn("Error decoding JSON", mock_st_error.call_args[0][0])

    @patch('st_app.st.error')
    @patch('builtins.open', side_effect=IOError("Disk read error"))
    def test_local_read_exception(self, mock_file_open, mock_st_error):
        uri = "/path/to/problematic/file.json"
        result = st_app.load_json_from_uri(uri)

        self.assertIsNone(result)
        mock_file_open.assert_called_once_with(uri, 'r')
        mock_st_error.assert_called_once_with(f"Error reading local file {uri}: Disk read error")

# TestRestaurantMergingLogic class is removed from here


class TestParseGcsUri(unittest.TestCase):

    def test_valid_uri_with_path(self):
        uri = "gs://bucket-name/path/to/blob/file.json"
        expected = ("bucket-name", "path/to/blob/file.json")
        self.assertEqual(_parse_gcs_uri(uri), expected)

    def test_valid_uri_no_path_just_filename(self):
        uri = "gs://bucket-name/file.json"
        expected = ("bucket-name", "file.json")
        self.assertEqual(_parse_gcs_uri(uri), expected)

    def test_invalid_uri_no_gs_prefix(self):
        uri = "http://bucket-name/blob"
        self.assertIsNone(_parse_gcs_uri(uri))
        
    def test_invalid_uri_gs_prefix_no_bucket_name(self):
        uri = "gs:///blob" # No bucket name
        self.assertIsNone(_parse_gcs_uri(uri))

    def test_invalid_uri_gs_prefix_no_blob_name_slash_only(self):
        uri = "gs://bucket-name/" # No blob name, just a slash
        self.assertIsNone(_parse_gcs_uri(uri))
        
    def test_invalid_uri_just_bucket_no_slash(self):
        uri = "gs://bucket-name" # No slash, so no blob name part
        self.assertIsNone(_parse_gcs_uri(uri))

    def test_invalid_uri_empty_string(self):
        uri = ""
        self.assertIsNone(_parse_gcs_uri(uri))

    def test_invalid_uri_gs_only(self):
        uri = "gs://"
        self.assertIsNone(_parse_gcs_uri(uri))


class TestLoadMasterData(unittest.TestCase):

    @patch('st_app.st.warning')
    @patch('st_app.st.success')
    @patch('st_app.st.info')
    @patch('st_app.load_json_from_uri') # Mocking the function passed as an argument
    def test_no_uri_provided(self, mock_load_json_func, mock_st_info, mock_st_success, mock_st_warning):
        result = load_master_data("", mock_load_json_func)
        self.assertEqual(result, [])
        mock_st_info.assert_called_once_with("No master restaurant data URI provided. Starting with empty master restaurant data.")
        mock_load_json_func.assert_not_called()
        mock_st_success.assert_not_called()
        mock_st_warning.assert_not_called()

    @patch('st_app.st.warning')
    @patch('st_app.st.success')
    @patch('st_app.st.info')
    @patch('st_app.load_json_from_uri')
    def test_load_json_returns_valid_list(self, mock_load_json_func, mock_st_info, mock_st_success, mock_st_warning):
        sample_list = [{'id': 1, 'name': 'Restaurant A'}, {'id': 2, 'name': 'Restaurant B'}]
        mock_load_json_func.return_value = sample_list
        uri = "dummy_uri"
        
        result = load_master_data(uri, mock_load_json_func)
        
        self.assertEqual(result, sample_list)
        mock_load_json_func.assert_called_once_with(uri)
        mock_st_success.assert_called_once_with(f"Successfully loaded master restaurant data with {len(sample_list)} records from {uri}.")
        mock_st_info.assert_not_called()
        mock_st_warning.assert_not_called()

    @patch('st_app.st.warning')
    @patch('st_app.st.success')
    @patch('st_app.st.info')
    @patch('st_app.load_json_from_uri')
    def test_load_json_returns_empty_list(self, mock_load_json_func, mock_st_info, mock_st_success, mock_st_warning):
        mock_load_json_func.return_value = []
        uri = "dummy_uri_empty"
        
        result = load_master_data(uri, mock_load_json_func)
        
        self.assertEqual(result, [])
        mock_load_json_func.assert_called_once_with(uri)
        mock_st_warning.assert_called_once_with(f"Master restaurant data loaded from {uri}, but it's empty.")
        mock_st_success.assert_not_called()
        mock_st_info.assert_not_called()

    @patch('st_app.st.warning')
    @patch('st_app.st.success')
    @patch('st_app.st.info')
    @patch('st_app.load_json_from_uri')
    def test_load_json_fails_returns_none(self, mock_load_json_func, mock_st_info, mock_st_success, mock_st_warning):
        mock_load_json_func.return_value = None
        uri = "dummy_uri_fail"
        
        result = load_master_data(uri, mock_load_json_func)
        
        self.assertEqual(result, [])
        mock_load_json_func.assert_called_once_with(uri)
        mock_st_warning.assert_called_once_with(f"Failed to load master restaurant data from {uri} (or it was empty/invalid). Proceeding with empty master restaurant data.")
        mock_st_success.assert_not_called()
        mock_st_info.assert_not_called()

    @patch('st_app.st.warning')
    @patch('st_app.st.success')
    @patch('st_app.st.info')
    @patch('st_app.load_json_from_uri')
    def test_load_json_returns_non_list_data(self, mock_load_json_func, mock_st_info, mock_st_success, mock_st_warning):
        non_list_data = {'data': 'unexpected format'}
        mock_load_json_func.return_value = non_list_data
        uri = "dummy_uri_non_list"
        
        result = load_master_data(uri, mock_load_json_func)
        
        self.assertEqual(result, [])
        mock_load_json_func.assert_called_once_with(uri)
        mock_st_warning.assert_called_once_with(f"Data loaded from {uri} is not in the expected list format. Type found: {type(non_list_data)}. Proceeding with empty master restaurant data.")
        mock_st_success.assert_not_called()
        mock_st_info.assert_not_called()


class TestProcessAndUpdateMasterData(unittest.TestCase):

    @patch('st_app.datetime') # Mock the datetime module in st_app
    @patch('st_app.st.warning')
    @patch('st_app.st.info')
    @patch('st_app.st.success')
    def test_empty_master_new_api_data(self, mock_st_success, mock_st_info, mock_st_warning, mock_datetime):
        fixed_now = dt(2023, 10, 26)
        fixed_date_str = "2023-10-26"
        mock_datetime.now.return_value = fixed_now # Configure the mock

        master_data = []
        api_data = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
                {"FHRSID": 1, "BusinessName": "Cafe One"},
                {"FHRSID": 2, "BusinessName": "Cafe Two"}
            ]}}
        }
        
        updated_data, new_count = process_and_update_master_data(master_data, api_data)
        
        self.assertEqual(new_count, 2)
        self.assertEqual(len(updated_data), 2)
        self.assertEqual(updated_data[0], {"FHRSID": 1, "BusinessName": "Cafe One", "first_seen": fixed_date_str})
        self.assertEqual(updated_data[1], {"FHRSID": 2, "BusinessName": "Cafe Two", "first_seen": fixed_date_str})
        mock_st_success.assert_called_once_with(f"Processed API response. Added {new_count} new restaurant records. Total unique records: {len(updated_data)}")
        mock_st_info.assert_not_called()
        mock_st_warning.assert_not_called()

    @patch('st_app.datetime')
    @patch('st_app.st.warning')
    @patch('st_app.st.info')
    @patch('st_app.st.success')
    def test_existing_master_new_and_overlapping_api(self, mock_st_success, mock_st_info, mock_st_warning, mock_datetime):
        fixed_now = dt(2023, 10, 26)
        fixed_date_str = "2023-10-26"
        mock_datetime.now.return_value = fixed_now

        master_data = [
            {"FHRSID": 1, "BusinessName": "Cafe One", "first_seen": "2023-01-01"},
            {"FHRSID": 3, "BusinessName": "Cafe Three", "first_seen": "2023-01-02"}
        ]
        api_data = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
                {"FHRSID": 1, "BusinessName": "Cafe One Updated"}, # Overlapping
                {"FHRSID": 2, "BusinessName": "Cafe Two New"},    # New
                {"FHRSID": 4, "BusinessName": "Cafe Four New"}    # New
            ]}}
        }
        
        updated_data, new_count = process_and_update_master_data(list(master_data), api_data) # Pass a copy
        
        self.assertEqual(new_count, 2)
        self.assertEqual(len(updated_data), 4)
        
        # Check existing entry not updated for first_seen, and business name is from original master
        self.assertEqual(updated_data[0], {"FHRSID": 1, "BusinessName": "Cafe One", "first_seen": "2023-01-01"})
        # Check original entry is still there
        self.assertEqual(updated_data[1], {"FHRSID": 3, "BusinessName": "Cafe Three", "first_seen": "2023-01-02"})
        # Check new entries
        self.assertTrue({"FHRSID": 2, "BusinessName": "Cafe Two New", "first_seen": fixed_date_str} in updated_data)
        self.assertTrue({"FHRSID": 4, "BusinessName": "Cafe Four New", "first_seen": fixed_date_str} in updated_data)
        mock_st_success.assert_called_once()

    @patch('st_app.datetime')
    @patch('st_app.st.warning')
    @patch('st_app.st.info')
    @patch('st_app.st.success')
    def test_api_data_empty_or_missing_details(self, mock_st_success, mock_st_info, mock_st_warning, mock_datetime):
        master_data_initial = [{"FHRSID": 1, "BusinessName": "Cafe One", "first_seen": "2023-01-01"}]

        scenarios = [
            ("Empty Detail List", {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': []}}}),
            ("Missing Detail Key", {'FHRSEstablishment': {'EstablishmentCollection': {}}}),
            ("Detail is None", {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': None}}}),
            ("Missing EstablishmentCollection", {'FHRSEstablishment': {}}),
            ("Missing FHRSEstablishment", {})
        ]

        for msg, api_data in scenarios:
            with self.subTest(msg=msg):
                mock_st_success.reset_mock()
                mock_st_info.reset_mock()
                mock_st_warning.reset_mock()
                
                updated_data, new_count = process_and_update_master_data(list(master_data_initial), api_data)
                
                self.assertEqual(new_count, 0, f"New count should be 0 for scenario: {msg}")
                self.assertEqual(updated_data, master_data_initial, f"Master data should not change for scenario: {msg}")
                
                if msg == "Empty Detail List":
                    mock_st_info.assert_called_once_with("API response contained no establishments in 'EstablishmentDetail'.")
                else: # Missing keys or None
                    mock_st_warning.assert_called_once_with("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
                mock_st_success.assert_called_once() # This success is for "Processed API response..."

    @patch('st_app.datetime')
    @patch('st_app.st.warning') # Not expecting warnings for this case
    @patch('st_app.st.info') # Not expecting info for this case
    @patch('st_app.st.success')
    def test_api_data_invalid_items(self, mock_st_success, mock_st_info, mock_st_warning, mock_datetime):
        fixed_now = dt(2023, 10, 26)
        fixed_date_str = "2023-10-26"
        mock_datetime.now.return_value = fixed_now

        master_data = []
        api_data = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
                {"FHRSID": 1, "BusinessName": "Valid Cafe"},
                "not a dictionary",  # Invalid item
                {"BusinessName": "Cafe Missing ID"}, # Missing FHRSID
                {"FHRSID": 2, "BusinessName": "Another Valid Cafe"},
                None # Another invalid item
            ]}}
        }
        
        updated_data, new_count = process_and_update_master_data(master_data, api_data)
        
        self.assertEqual(new_count, 2)
        self.assertEqual(len(updated_data), 2)
        self.assertTrue({"FHRSID": 1, "BusinessName": "Valid Cafe", "first_seen": fixed_date_str} in updated_data)
        self.assertTrue({"FHRSID": 2, "BusinessName": "Another Valid Cafe", "first_seen": fixed_date_str} in updated_data)
        mock_st_success.assert_called_once()
        mock_st_info.assert_not_called()
        mock_st_warning.assert_not_called()



class TestUploadToGcs(unittest.TestCase):

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    @patch('st_app._parse_gcs_uri') # Mock _parse_gcs_uri from st_app module
    def test_successful_upload(self, mock_parse_uri, mock_gcs_client, mock_st_error):
        mock_parse_uri.return_value = ("test-bucket", "test-dir/test-blob.json")
        
        mock_blob_instance = MagicMock()
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob_instance
        
        mock_gcs_client_instance = MagicMock()
        mock_gcs_client_instance.bucket.return_value = mock_bucket_instance
        mock_gcs_client.return_value = mock_gcs_client_instance

        gcs_uri = "gs://test-bucket/test-dir/test-blob.json"
        data_string = '{"data": "test content"}'
        content_type = "application/json"

        result = upload_to_gcs(gcs_uri, data_string, content_type)

        self.assertTrue(result)
        mock_parse_uri.assert_called_once_with(gcs_uri)
        mock_gcs_client.assert_called_once()
        mock_gcs_client_instance.bucket.assert_called_once_with("test-bucket")
        mock_bucket_instance.blob.assert_called_once_with("test-dir/test-blob.json")
        mock_blob_instance.upload_from_string.assert_called_once_with(data_string, content_type=content_type)
        mock_st_error.assert_not_called()

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    @patch('st_app._parse_gcs_uri')
    def test_upload_failure_invalid_uri(self, mock_parse_uri, mock_gcs_client, mock_st_error):
        invalid_gcs_uri = "not-a-gs-uri/test-blob.json"
        mock_parse_uri.return_value = None # Simulate _parse_gcs_uri failing

        data_string = '{"data": "test content"}'
        
        result = upload_to_gcs(invalid_gcs_uri, data_string)

        self.assertFalse(result)
        mock_parse_uri.assert_called_once_with(invalid_gcs_uri)
        mock_st_error.assert_called_once_with(f"Invalid GCS URI format: {invalid_gcs_uri}. It must start with gs:// and include a bucket and blob name.")
        mock_gcs_client.assert_not_called() # Client should not be called if parsing fails

    @patch('st_app.st.error')
    @patch('st_app.storage.Client')
    @patch('st_app._parse_gcs_uri')
    def test_upload_failure_gcs_exception(self, mock_parse_uri, mock_gcs_client, mock_st_error):
        gcs_uri = "gs://test-bucket/forbidden-blob.json"
        mock_parse_uri.return_value = ("test-bucket", "forbidden-blob.json")

        mock_blob_instance = MagicMock()
        # Simulate an exception during upload, e.g., Forbidden
        mock_blob_instance.upload_from_string.side_effect = st_app.exceptions.Forbidden("Permission Denied on GCS")
        
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob_instance
        
        mock_gcs_client_instance = MagicMock()
        mock_gcs_client_instance.bucket.return_value = mock_bucket_instance
        mock_gcs_client.return_value = mock_gcs_client_instance
        
        data_string = '{"data": "sensitive content"}'
        result = upload_to_gcs(gcs_uri, data_string)

        self.assertFalse(result)
        mock_parse_uri.assert_called_once_with(gcs_uri)
        mock_gcs_client.assert_called_once()
        mock_blob_instance.upload_from_string.assert_called_once()
        mock_st_error.assert_called_once()
        # Check if the error message contains the expected parts
        self.assertIn(f"Error uploading data to GCS ({gcs_uri})", mock_st_error.call_args[0][0])
        self.assertIn("Permission Denied on GCS", mock_st_error.call_args[0][0])


class TestFetchApiData(unittest.TestCase):

    @patch('st_app.st.error')
    @patch('st_app.requests.get')
    def test_successful_api_call(self, mock_requests_get, mock_st_error):
        expected_data = {"FHRSID": 123, "BusinessName": "Test Cafe"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_data
        mock_requests_get.return_value = mock_response

        longitude, latitude = -0.123, 51.456
        max_results = 150  # Added max_results
        result = fetch_api_data(longitude, latitude, max_results)

        self.assertEqual(result, expected_data)
        # Updated URL assertion
        mock_requests_get.assert_called_once_with(f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/{longitude}/{latitude}/1/{max_results}/json")
        mock_st_error.assert_not_called()

    @patch('st_app.st.error')
    @patch('st_app.requests.get')
    def test_api_call_network_error(self, mock_requests_get, mock_st_error):
        # Using requests.exceptions.RequestException from the global requests import
        mock_requests_get.side_effect = requests.exceptions.RequestException("Network connection failed")

        longitude, latitude = 0.5, 50.5
        max_results = 150 # Added max_results
        result = fetch_api_data(longitude, latitude, max_results)

        self.assertIsNone(result)
        mock_requests_get.assert_called_once()
        mock_st_error.assert_called_once()
        self.assertIn("An exception occurred while making the API request: Network connection failed", mock_st_error.call_args[0][0])

    @patch('st_app.st.error')
    @patch('st_app.requests.get')
    def test_api_call_non_200_status(self, mock_requests_get, mock_st_error):
        mock_response = MagicMock()
        mock_response.status_code = 404
        # .json() might not be called or might raise an error for non-200, 
        # but fetch_api_data currently tries to call .json() only if status is 200.
        # So, no need to mock .json() for this case.
        mock_requests_get.return_value = mock_response

        longitude, latitude = 1.0, 49.0
        max_results = 150 # Added max_results
        result = fetch_api_data(longitude, latitude, max_results)

        self.assertIsNone(result)
        mock_requests_get.assert_called_once()
        mock_st_error.assert_called_once_with("Error: Could not fetch data from the API. Status Code: 404")

# Helper class for mock responses in test_fetch_data_multiple_coordinates
class MockResponseHelper:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

class TestHandleFetchDataAction(unittest.TestCase):

    @patch('st_app.st.text_area') # Mocked, but value passed directly to handle_fetch_data_action
    @patch('st_app.st.number_input') # Mocked, but value passed directly
    @patch('st_app.st.text_input') # Mocked, but values passed directly
    @patch('st_app.requests.get')
    @patch('st_app.load_master_data')
    # process_and_update_master_data is NOT mocked to test its integration
    @patch('st_app.upload_to_gcs')
    @patch('st_app.write_to_bigquery')
    @patch('st_app.display_data') # Mock display_data to avoid actual UI rendering
    @patch('st_app.st.info')
    @patch('st_app.st.success')
    @patch('st_app.st.warning')
    @patch('st_app.st.error')
    @patch('st_app.st.write')
    @patch('st_app.st.stop') # Mock st.stop to prevent test execution halt
    def test_fetch_data_multiple_coordinates(
        self, mock_st_stop, mock_st_write, mock_st_error, mock_st_warning, 
        mock_st_success, mock_st_info, mock_display_data,
        mock_write_to_bigquery, mock_upload_to_gcs, mock_load_master_data,
        mock_requests_get, mock_st_text_input_global, 
        mock_st_number_input_global, mock_st_text_area_global 
        # Renamed global mocks to avoid clash if st_app.py also had variables with these names
    ):
        # --- Configure Mock Return Values (even if values passed directly, mocks might be checked by st_app) ---
        
        # For st_app.py's global scope st.text_area, st.number_input, st.text_input calls
        # These are not directly used by handle_fetch_data_action parameters but are in st_app.py
        mock_st_text_area_global.return_value = "1.0,2.0\n-3.5,4.8" # Default for global call
        mock_st_number_input_global.return_value = 200 # Default for global call
        mock_st_text_input_global.return_value = "gs://dummy_global_uri/" # Default for global calls


        # Values to be passed directly to handle_fetch_data_action
        coordinate_pairs_value = "1.0,2.0\n-3.5,4.8"
        max_results_value = 150
        gcs_destination_uri_value = "gs://test-gcs-destination-folder/"
        master_list_uri_value = "gs://test-master-list-uri/master.json"
        gcs_master_output_uri_value = "gs://test-gcs-master-output/output.json"
        bq_full_path_value = "test_project.test_dataset.test_table"

        # Mock requests.get for API calls
        mock_response_1_data = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [{'FHRSID': 1, 'BusinessName': 'Restaurant A (Coords 1,2)'}]
                }
            }
        }
        mock_response_2_data = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [
                        {'FHRSID': 2, 'BusinessName': 'Restaurant B (Coords -3.5,4.8)'},
                        {'FHRSID': 3, 'BusinessName': 'Restaurant C (Coords -3.5,4.8)'}
                    ]
                }
            }
        }
        mock_requests_get.side_effect = [
            MockResponseHelper(mock_response_1_data, 200),
            MockResponseHelper(mock_response_2_data, 200)
        ]

        mock_load_master_data.return_value = [] 
        mock_upload_to_gcs.return_value = True
        mock_write_to_bigquery.return_value = True
        
        # --- Call the refactored function ---
        result_master_data = st_app.handle_fetch_data_action(
            coordinate_pairs_str=coordinate_pairs_value,
            max_results=max_results_value,
            gcs_destination_uri_str=gcs_destination_uri_value,
            master_list_uri_str=master_list_uri_value,
            gcs_master_output_uri_str=gcs_master_output_uri_value,
            bq_full_path_str=bq_full_path_value
        )

        # --- Assertions ---
        expected_api_calls = [
            unittest.mock.call(f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/1.0/2.0/1/{max_results_value}/json"),
            unittest.mock.call(f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/-3.5/4.8/1/{max_results_value}/json")
        ]
        mock_requests_get.assert_has_calls(expected_api_calls, any_order=False)

        mock_load_master_data.assert_called_once_with(master_list_uri_value, st_app.load_json_from_uri)
        
        self.assertIsNotNone(result_master_data)
        self.assertEqual(len(result_master_data), 3) 
        fhrsid_list = sorted([item['FHRSID'] for item in result_master_data])
        self.assertEqual(fhrsid_list, [1, 2, 3])
        for item in result_master_data:
            self.assertIn('first_seen', item) # Checked because real process_and_update_master_data runs

        self.assertEqual(mock_upload_to_gcs.call_count, 2)
        args_raw_upload, _ = mock_upload_to_gcs.call_args_list[0]
        self.assertTrue(args_raw_upload[0].startswith(gcs_destination_uri_value + "combined_api_response_"))
        uploaded_raw_data = json.loads(args_raw_upload[1])
        self.assertEqual(len(uploaded_raw_data['FHRSEstablishment']['EstablishmentCollection']['EstablishmentDetail']), 3)

        args_master_upload, _ = mock_upload_to_gcs.call_args_list[1]
        self.assertEqual(args_master_upload[0], gcs_master_output_uri_value)
        uploaded_master_data = json.loads(args_master_upload[1])
        self.assertEqual(len(uploaded_master_data), 3)

        mock_write_to_bigquery.assert_called_once()
        # Check the DataFrame passed to write_to_bigquery
        call_args_bq, _ = mock_write_to_bigquery.call_args
        df_passed_to_bq = call_args_bq[0]
        self.assertIsInstance(df_passed_to_bq, pd.DataFrame)
        self.assertEqual(len(df_passed_to_bq), 3)
        self.assertEqual(sorted(list(df_passed_to_bq['FHRSID'])), [1,2,3])


        mock_st_info.assert_any_call("Found 2 valid coordinate pairs. Fetching data for each...")
        mock_st_success.assert_any_call("Total establishments fetched from all API calls: 3")
        # This success call comes from process_and_update_master_data
        mock_st_success.assert_any_call("Processed API response. Added 3 new restaurant records. Total unique records: 3")
        # This success call comes from upload_to_gcs (twice) and write_to_bigquery
        self.assertGreaterEqual(mock_st_success.call_count, 3)


class TestAppMainLogic(unittest.TestCase):
    # Keep a reference to the original st module
    original_st = st_app.st

    def setUp(self):
        # Reset mocks for st specific functions for each test
        # This ensures that st.info, st.warning etc. are fresh mocks for each test
        st_app.st = MagicMock()
        # Mock specific streamlit functions that are called directly in the app
        st_app.st.title = MagicMock()
        st_app.st.number_input = MagicMock()
        st_app.st.text_input = MagicMock()
        st_app.st.button = MagicMock()
        st_app.st.info = MagicMock()
        st_app.st.warning = MagicMock()
        st_app.st.error = MagicMock() # In case any other part of the script calls it
        st_app.st.success = MagicMock() # For GCS/BQ uploads
        st_app.st.dataframe = MagicMock() # For display_data
        st_app.st.json = MagicMock() # For display_data fallback

    def tearDown(self):
        # Restore the original st module to avoid interference between test classes
        st_app.st = self.original_st
        # Reset all mocks on the original st module if necessary,
        # but individual mocks are preferred.
        # For now, we'll rely on the per-test setup.

    # _run_app_script_main_logic and the tests that use it might need to be adapted
    # if the global st_app.py structure changed significantly due to refactoring for
    # handle_fetch_data_action. For now, assuming they test other parts or are separate.
    def _run_app_script_main_logic(self):
        # This helper will simulate the execution of the main part of st_app.py
        # It relies on mocks for input functions (like st.button) to guide execution.
        # We need to ensure that all global-level st calls in st_app.py are also mocked
        # if they exist and are problematic. The st_app.st = MagicMock() in setUp
        # should handle this for st.* calls within the script.

        # The main logic is inside `if st.button("Fetch Data")`
        # So, we can simulate this by calling the relevant parts of the script.
        # A robust way is to put the main app logic into a function in st_app.py.
        # Since we can't change st_app.py, we'll execute the script.
        # We need to be careful as this re-runs the entire script including imports.
        
        # To avoid issues with re-importing and to control the execution flow better,
        # we will directly call the code block that is under `if st.button("Fetch Data")`
        # This requires careful mocking of functions called before this block.
        # For this specific case, the relevant code is already in st_app.py
        # and will be executed if st.button returns True.

        # We'll use a simplified approach: directly use the current st_app module
        # and rely on its state after mocks have been applied.
        # This means the `if st.button("Fetch Data"):` block will be tested
        # by ensuring `st_app.st.button` returns True and then checking effects.

        # Simulate input values. These will be returned by the mocked st.number_input and st.text_input
        # The order of calls to st.number_input matters.
        # 1. longitude, 2. latitude, 3. max_results_input
        # Other inputs like GCS URIs will be mocked to return empty strings or None
        # to simplify tests focusing on the API result count logic.

        # Default mock setup for inputs to avoid None values if not specified by the test
        def number_input_side_effect(label, **kwargs):
            if "Longitude" in label: return 0.0
            if "Latitude" in label: return 0.0
            if "Max Results" in label: return 100 # Default, can be overridden by specific tests
            return 0 # Default for any other number input
        
        st_app.st.number_input.side_effect = number_input_side_effect
        st_app.st.text_input.return_value = "" # Default for all text inputs

        # Simulate button press
        st_app.st.button.return_value = True

        # Now, we need to trigger the execution of the code block under `if st.button("Fetch Data")`.
        # One way to do this without `exec` is to have a function in `st_app.py` that contains this block.
        # Since that's not the case, we'll rely on the fact that when `test_st_app.py` imports `st_app`,
        # the `if st.button("Fetch Data")` block is defined.
        # We can call it by re-evaluating that part of the module or by running the module's main execution path.
        
        # Let's try to execute the script's content in a controlled manner.
        # We'll use a global variable in st_app to store the app's main function if it were refactored.
        # For now, we assume that by setting st.button to True, and then importing/running st_app,
        # the logic will execute.
        # The import `import st_app` at the top of test_st_app.py already executes the script once.
        # Re-running it via `exec` or `importlib.reload` is tricky with mocks.

        # A simpler simulation:
        # The `if st.button("Fetch Data")` block in st_app.py is at the global level.
        # When st_app is imported, this block is defined.
        # If st.button is mocked to return True *before* this block is encountered by Python interpreter,
        # then the code inside it would run. However, imports happen first.
        # So, we need to re-evaluate the part of st_app.py that contains this conditional block.

        # Let's try a slightly different approach for triggering the logic:
        # We will patch all inputs, then we can "re-run" the script's main definition
        # by using `importlib.reload`. This is generally safer than exec.
        # However, it can have side effects with mocks if not handled carefully.

        # For now, let's assume the initial import of st_app in test_st_app.py
        # already sets up the UI elements. We then trigger the "Fetch Data" button.
        # The key is that the `if st.button("Fetch Data"):` block is part of the script's main execution path.
        # So, if we can re-trigger that path after setting mocks, it should work.

        # The easiest way, given the current structure and the global `st_app.st = MagicMock()`
        # is to encapsulate the main logic of `st_app.py` (from `st.title` onwards)
        # into a function, then call that function. Since we cannot modify `st_app.py`,
        # we will simulate the relevant section.
        
        # The Streamlit app's structure means the code from st.title() downwards is run on import or re-run.
        # We'll assume the `if st.button("Fetch Data")` part is what we need to "re-run"
        # The mocks should already be in place.
        
        # Let's refine: the `if st.button("Fetch Data")` block in `st_app.py` uses the *current* values
        # of `longitude`, `latitude`, `max_results_input` etc., which are defined by `st.number_input` calls
        # *before* the button.
        # So, we need to ensure these `st.number_input` mocks are set up correctly.

        # The actual execution of the `if st.button("Fetch Data"):` block happens when st_app.py is run.
        # Our mocks for st.button, st.number_input etc. need to be in place when this happens.
        # The import at the top of test_st_app.py runs st_app.py once.
        # We can use importlib.reload(st_app) to re-run it after setting up mocks for a specific test.
        
        # This requires careful handling of mocks so they are active during the reload.
        # Patching at the class or method level should ensure this.

        # Let's execute the main part of st_app.py.
        # All functions in st_app are defined when st_app is imported.
        # The UI rendering and logic execution happens from `st.title(...)` downwards.
        # We can simulate this by calling a hypothetical main function.
        # Since it doesn't exist, we'll have to be more direct.

        # The `if st.button("Fetch Data")` block:
        # We assume our mocks for st.number_input, st.text_input, and st.button are active.
        # Then we call the function that uses these inputs, which is `fetch_api_data`
        # and the subsequent logic.
        
        # Simplified approach for this specific test:
        # The code inside `if st.button("Fetch Data")` is what we care about.
        # We will mock the inputs (`longitude`, `latitude`, `max_results_input`) that are read *before* this block.
        # Then we will call the sequence of functions as they appear in the `if api_data:` block.
        
        # This is still not ideal. The most direct way to test the `if st.button` block
        # is to ensure st.button() returns true and then the script executes that path.
        # This happens upon import/reload if the conditions are met.

        # Let's assume the test structure will mock `st.button` to return `True`.
        # Then, the code inside `if st.button("Fetch Data")` will be executed.
        # We need to ensure `fetch_api_data` is mocked correctly for each test case.
        
        # The `st_app.py` script will be executed (or reloaded).
        # The critical part is that `st.number_input` calls must be mocked before `max_results_input` is used.
        # And `fetch_api_data` must be mocked before it's called.

        # Let's try to execute the relevant part of the script directly.
        # This is fragile. A better way is to refactor st_app.py.
        # For now, we will mock inputs, then call the sequence of operations
        # that would happen inside the `if st.button("Fetch Data")` and `if api_data:` blocks.
        
        # No, this is also not quite right. We need to test the Streamlit control flow.
        # The `exec` approach or `importlib.reload` is probably necessary if we can't refactor.
        
        # Let's use `importlib.reload` for now. It's cleaner than `exec`.
        import importlib
        # All patches should be active when reload happens.
        # The `st_app.st` mock needs to be correctly configured.
        
        # Mock other functions that are called within the button click logic
        # to avoid side effects not relevant to these specific tests.
        st_app.load_master_data = MagicMock(return_value=[]) # Assume empty master data
        st_app.process_and_update_master_data = MagicMock(return_value=([], 0))
        st_app.upload_to_gcs = MagicMock(return_value=True)
        st_app.display_data = MagicMock()
        st_app.write_to_bigquery = MagicMock(return_value=True)
        st_app.pd.json_normalize = MagicMock(return_value=pd.DataFrame()) # Mock pandas normalization

        importlib.reload(st_app) # This will re-run st_app.py with current mocks


    @patch('st_app.fetch_api_data')
    def test_results_less_than_max(self, mock_fetch_api_data):
        max_val = 100
        returned_val = 50
        
        # Mock st.number_input to return our max_val for the 'Max Results' input
        def number_input_side_effect(label, **kwargs):
            if "Max Results" in label: return max_val
            if "Longitude" in label: return 0.0
            if "Latitude" in label: return 0.0
            return 0
        st_app.st.number_input.side_effect = number_input_side_effect

        mock_fetch_api_data.return_value = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [{} for _ in range(returned_val)]}}
        }
        
        self._run_app_script_main_logic()

        st_app.st.info.assert_any_call(f"Number of results returned by API: {returned_val}")
        # Ensure warning is not called for this case
        # Check all calls to st.warning, make sure none match the specific cap warning.
        for call_args in st_app.st.warning.call_args_list:
            self.assertNotIn("results, which matches the `max_results` input", call_args[0][0])


    @patch('st_app.fetch_api_data')
    def test_results_equal_to_max_cap_warning(self, mock_fetch_api_data):
        max_val = 75
        returned_val = 75

        def number_input_side_effect(label, **kwargs):
            if "Max Results" in label: return max_val
            if "Longitude" in label: return 0.0
            if "Latitude" in label: return 0.0
            return 0
        st_app.st.number_input.side_effect = number_input_side_effect

        mock_fetch_api_data.return_value = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [{} for _ in range(returned_val)]}}
        }

        self._run_app_script_main_logic()

        st_app.st.info.assert_any_call(f"Number of results returned by API: {returned_val}")
        st_app.st.warning.assert_any_call(
            f"Warning: The API returned {returned_val} results, which matches the `max_results` input. "
            "This might indicate the results are capped. Consider increasing the 'Max Results for API Call' "
            "value if you suspect more data is available."
        )

    @patch('st_app.fetch_api_data')
    def test_api_returns_no_establishments(self, mock_fetch_api_data):
        max_val = 100
        returned_val = 0

        def number_input_side_effect(label, **kwargs):
            if "Max Results" in label: return max_val
            if "Longitude" in label: return 0.0
            if "Latitude" in label: return 0.0
            return 0
        st_app.st.number_input.side_effect = number_input_side_effect
        
        # Scenario 1: EstablishmentDetail is an empty list
        mock_fetch_api_data.return_value = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': []}}}
        
        self._run_app_script_main_logic()
        st_app.st.info.assert_any_call(f"Number of results returned by API: {returned_val}")
        for call_args in st_app.st.warning.call_args_list:
            self.assertNotIn("results, which matches the `max_results` input", call_args[0][0])
        
        st_app.st.info.reset_mock()
        st_app.st.warning.reset_mock()

        # Scenario 2: EstablishmentDetail is None (explicitly null in JSON)
        mock_fetch_api_data.return_value = {
            'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': None}}}
        
        self._run_app_script_main_logic()
        st_app.st.info.assert_any_call(f"Number of results returned by API: {returned_val}") # len([]) is 0
        for call_args in st_app.st.warning.call_args_list:
            self.assertNotIn("results, which matches the `max_results` input", call_args[0][0])

        st_app.st.info.reset_mock()
        st_app.st.warning.reset_mock()

        # Scenario 3: EstablishmentCollection is missing (or FHRSEstablishment)
        mock_fetch_api_data.return_value = {'FHRSEstablishment': {}}
        self._run_app_script_main_logic()
        st_app.st.info.assert_any_call(f"Number of results returned by API: {returned_val}") # .get('EstablishmentDetail', [])
        for call_args in st_app.st.warning.call_args_list:
            self.assertNotIn("results, which matches the `max_results` input", call_args[0][0])


class TestBigQueryFunctions(unittest.TestCase):

    @patch('st_app.bigquery.Client') # Mock the BigQuery client in st_app
    @patch('st_app.st') # Mock Streamlit methods like st.success, st.error
    def test_write_to_bigquery_success(self, mock_st, mock_bq_client_constructor):
        # Arrange
        mock_bq_client_instance = MagicMock()
        mock_bq_client_constructor.return_value = mock_bq_client_instance
        
        sample_data = [{'col1': 'data1', 'col2': 1}, {'col1': 'data2', 'col2': 2}]
        sample_df = pd.DataFrame(sample_data)
        
        project_id = "test_project"
        dataset_id = "test_dataset"
        table_id = "test_table"
        expected_table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

        # Act
        result = write_to_bigquery(sample_df, project_id, dataset_id, table_id)

        # Assert
        mock_bq_client_constructor.assert_called_once_with(project=project_id)
        
        # Check that load_table_from_dataframe was called
        call_args = mock_bq_client_instance.load_table_from_dataframe.call_args
        self.assertIsNotNone(call_args, "load_table_from_dataframe was not called")

        # Check the arguments of load_table_from_dataframe
        loaded_df = call_args[0][0]
        loaded_table_ref = call_args[0][1]
        job_config = call_args[1]['job_config']

        pd.testing.assert_frame_equal(loaded_df, sample_df)
        self.assertEqual(loaded_table_ref, expected_table_ref_str)
        self.assertEqual(job_config.write_disposition, bigquery.WriteDisposition.WRITE_TRUNCATE)
        
        mock_bq_client_instance.load_table_from_dataframe.return_value.result.assert_called_once() # Check job.result() was called
        mock_st.success.assert_called_once()
        self.assertTrue(result)

    @patch('st_app.bigquery.Client')
    @patch('st_app.st')
    def test_write_to_bigquery_failure(self, mock_st, mock_bq_client_constructor):
        # Arrange
        mock_bq_client_instance = MagicMock()
        mock_bq_client_constructor.return_value = mock_bq_client_instance
        mock_bq_client_instance.load_table_from_dataframe.side_effect = Exception("Test BQ Error")

        sample_df = pd.DataFrame([{'col1': 'data1'}])
        project_id = "test_project"
        dataset_id = "test_dataset"
        table_id = "test_table"
        expected_table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

        # Act
        result = write_to_bigquery(sample_df, project_id, dataset_id, table_id)

        # Assert
        mock_bq_client_constructor.assert_called_once_with(project=project_id)
        mock_bq_client_instance.load_table_from_dataframe.assert_called_once()
        mock_st.error.assert_called_once_with(f"Error writing data to BigQuery table {expected_table_ref_str}: Test BQ Error")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
