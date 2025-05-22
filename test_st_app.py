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
    load_master_data, load_json_from_uri, process_and_update_master_data
)


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
        result = fetch_api_data(longitude, latitude)

        self.assertEqual(result, expected_data)
        mock_requests_get.assert_called_once_with(f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/{longitude}/{latitude}/1/500/json")
        mock_st_error.assert_not_called()

    @patch('st_app.st.error')
    @patch('st_app.requests.get')
    def test_api_call_network_error(self, mock_requests_get, mock_st_error):
        # Using requests.exceptions.RequestException from the global requests import
        mock_requests_get.side_effect = requests.exceptions.RequestException("Network connection failed")

        longitude, latitude = 0.5, 50.5
        result = fetch_api_data(longitude, latitude)

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
        result = fetch_api_data(longitude, latitude)

        self.assertIsNone(result)
        mock_requests_get.assert_called_once()
        mock_st_error.assert_called_once_with("Error: Could not fetch data from the API. Status Code: 404")
