import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
from datetime import datetime as dt # Alias to avoid conflict with datetime module in st_app
import pandas as pd

# Import the Streamlit app module to be tested
import st_app

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
    @patch('st_app.storage.Client') # Still need to mock client even if not used for this path
    def test_gcs_no_blob_name(self, mock_gcs_client, mock_st_error):
        uri = "gs://test-bucket/" # URI pointing to bucket only
        result = st_app.load_json_from_uri(uri)
        self.assertIsNone(result)
        mock_st_error.assert_called_once_with("Invalid GCS URI: No file specified in gs://test-bucket/")
        mock_gcs_client.assert_called_once() # Client is initialized before check

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


class TestRestaurantMergingLogic(unittest.TestCase):
    
    def setUp(self):
        # Mock all relevant st functions that could be called within the "Fetch Data" block
        self.mock_st_info = patch('st_app.st.info').start()
        self.mock_st_success = patch('st_app.st.success').start()
        self.mock_st_warning = patch('st_app.st.warning').start()
        self.mock_st_error = patch('st_app.st.error').start()
        self.mock_st_dataframe = patch('st_app.st.dataframe').start()
        self.mock_st_json = patch('st_app.st.json').start() # For fallback display

        # Mock external calls
        self.mock_requests_get = patch('st_app.requests.get').start()
        self.mock_load_json_uri = patch('st_app.load_json_from_uri').start()
        self.mock_pd_normalize = patch('st_app.pd.json_normalize').start()
        
        # Mock datetime
        self.mock_datetime_now = patch('st_app.datetime').start() # Patch the whole module in st_app
        self.fixed_date = dt(2024, 1, 15)
        self.fixed_date_str = "2024-01-15"
        self.mock_datetime_now.now.return_value = self.fixed_date
        
        # Mock GCS upload related parts
        mock_gcs_client_constructor = patch('st_app.storage.Client').start()
        self.mock_gcs_client_instance = MagicMock()
        self.mock_gcs_bucket_instance = MagicMock()
        self.mock_gcs_blob_instance = MagicMock()
        mock_gcs_client_constructor.return_value = self.mock_gcs_client_instance
        self.mock_gcs_client_instance.bucket.return_value = self.mock_gcs_bucket_instance
        self.mock_gcs_bucket_instance.blob.return_value = self.mock_gcs_blob_instance
        self.addCleanup(mock_gcs_client_constructor.stop) # Ensure this patch is stopped

        # Simulate inputs that are typically read from st.text_input
        # These are module-level in st_app.py, so we patch them there
        # We use self.addCleanup to ensure these are stopped after each test
        # This is better than managing start/stop manually or using a class-level patcher for instance-specific values
        self.patcher_master_restaurant_data_uri = patch('st_app.master_list_uri', "dummy_master_uri") # Patches st_app.master_list_uri
        self.patcher_gcs_destination_uri = patch('st_app.gcs_destination_uri', "") # Default to no GCS for most tests
        self.patcher_gcs_master_data_output_uri = patch('st_app.gcs_master_dictionary_output_uri', "") # Patches st_app.gcs_master_dictionary_output_uri

        self.st_app_master_list_uri = self.patcher_master_restaurant_data_uri.start()
        self.st_app_gcs_destination_uri = self.patcher_gcs_destination_uri.start()
        self.st_app_gcs_master_data_output_uri = self.patcher_gcs_master_data_output_uri.start()
        
        self.addCleanup(self.patcher_master_restaurant_data_uri.stop)
        self.addCleanup(self.patcher_gcs_destination_uri.stop)
        self.addCleanup(self.patcher_gcs_master_data_output_uri.stop)


    def tearDown(self):
        # patch.stopall() # Not strictly necessary if using addCleanup for all patches started in setUp
        # However, if any patch is started with .start() and not added to addCleanup,
        # or if a test method itself uses .start(), then patch.stopall() is a good safety net.
        # For now, relying on addCleanup for specific patches.
        pass # Relying on addCleanup

    def _run_fetch_data_logic(self, current_master_list_uri=None): # Removed current_gcs_destination_uri
        # Helper to simulate the core logic of the "Fetch Data" button in st_app.py
        # Patched st_app.gcs_destination_uri and st_app.gcs_master_dictionary_output_uri
        # should be set directly in test methods before calling this helper.

        if current_master_list_uri is not None:
            st_app.master_list_uri = current_master_list_uri
        
        # --- Replicated logic from st_app.py ---

        # 1.a. Load Master Restaurant Data
        master_restaurant_data = []
        if st_app.master_list_uri:
            loaded_data = self.mock_load_json_uri(st_app.master_list_uri)
            if loaded_data is not None:
                if isinstance(loaded_data, list):
                    master_restaurant_data = loaded_data
                    if master_restaurant_data:
                        self.mock_st_success(f"Successfully loaded master restaurant data with {len(master_restaurant_data)} records from {st_app.master_list_uri}.")
                    else:
                        self.mock_st_warning(f"Master restaurant data loaded from {st_app.master_list_uri}, but it's empty.")
                else:
                    self.mock_st_warning(f"Data loaded from {st_app.master_list_uri} is not in the expected format (e.g., a list of records). Type found: {type(loaded_data)}. Proceeding with empty master restaurant data.")
                    master_restaurant_data = []
            else:
                self.mock_st_warning(f"Failed to load master restaurant data from {st_app.master_list_uri} (or it was empty/invalid). Proceeding with empty master restaurant data.")
                master_restaurant_data = []
        else:
            self.mock_st_info("No master restaurant data URI provided. Starting with empty master restaurant data.")
            master_restaurant_data = []

        # 1.b. Process API Response and Update Master Restaurant Data
        api_data_response = self.mock_requests_get() # Mocked requests.get
        api_data_response.raise_for_status() # Simulate check or assume 200 for tests using this helper
        api_json_content = api_data_response.json() # This is the raw API data

        api_establishments = api_json_content.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
        if api_establishments is None: 
            api_establishments = []

        existing_fhrsid_set = {est['FHRSID'] for est in master_restaurant_data if isinstance(est, dict) and 'FHRSID' in est}
        today_date_str = self.mock_datetime_now.now().strftime("%Y-%m-%d") # Uses mocked datetime from setUp
        new_restaurants_added_count = 0

        for api_establishment in api_establishments:
            if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
                if api_establishment['FHRSID'] not in existing_fhrsid_set:
                    api_establishment['first_seen'] = today_date_str
                    master_restaurant_data.append(api_establishment)
                    existing_fhrsid_set.add(api_establishment['FHRSID'])
                    new_restaurants_added_count += 1
        
        self.mock_st_success(f"Processed API response. Added {new_restaurants_added_count} new restaurant records. Total unique records: {len(master_restaurant_data)}")

        # GCS Upload Logic (New Two-Part Logic)
        # Part 1: Upload Raw API Response
        if st_app.gcs_destination_uri: # Patched by test method
            if not st_app.gcs_destination_uri.startswith("gs://"):
                self.mock_st_error("Invalid GCS URI for API Response. It must start with gs://")
            else:
                try:
                    # self.mock_gcs_client_instance is from setUp
                    current_date = self.mock_datetime_now.now().strftime("%Y-%m-%d")
                    api_response_filename = f"api_response_{current_date}.json"
                    
                    bucket_name_api = st_app.gcs_destination_uri.split("/")[2]
                    folder_path_api = "/".join(st_app.gcs_destination_uri.split("/")[3:])
                    
                    if folder_path_api and not folder_path_api.endswith('/'):
                        folder_path_api += '/'
                    
                    blob_path_api = f"{folder_path_api}{api_response_filename}"
                    if blob_path_api.startswith('/'): # Handle root bucket case
                       blob_path_api = blob_path_api[1:]

                    # self.mock_gcs_bucket_instance.blob will be called here
                    # bucket_api = self.mock_gcs_client_instance.bucket(bucket_name_api)
                    # blob_api = bucket_api.blob(blob_path_api) # This call will be asserted
                    
                    # For testing, we use the pre-mocked instances from setUp.
                    # The specific bucket name and blob path will be checked in assertions.
                    self.mock_gcs_client_instance.bucket(bucket_name_api) # Call to ensure it's tracked if needed
                    self.mock_gcs_bucket_instance.blob(blob_path_api) # Call to ensure it's tracked

                    self.mock_gcs_blob_instance.upload_from_string(
                        json.dumps(api_json_content, indent=4), 
                        content_type='application/json'
                    )
                    self.mock_st_success(f"Successfully uploaded raw API response to gs://{bucket_name_api}/{blob_path_api}")
                except Exception as e:
                    self.mock_st_error(f"Error uploading raw API response to GCS: {e}")
        
        # Part 2: Upload Master Restaurant Data
        if st_app.gcs_master_dictionary_output_uri: # Patched by test method (st_app.gcs_master_dictionary_output_uri)
            if not st_app.gcs_master_dictionary_output_uri.startswith("gs://"):
                self.mock_st_error("Invalid GCS URI for Master Restaurant Data. It must start with gs://") # Updated error message
            else:
                try:
                    # self.mock_gcs_client_instance is from setUp
                    bucket_name_master = st_app.gcs_master_dictionary_output_uri.split("/")[2]
                    blob_name_master = "/".join(st_app.gcs_master_dictionary_output_uri.split("/")[3:])

                    if not blob_name_master: 
                        self.mock_st_error("Invalid GCS URI for Master Restaurant Data: File name is missing.") # Updated error message
                    else:
                        # self.mock_gcs_bucket_instance.blob will be called here again
                        # bucket_master = self.mock_gcs_client_instance.bucket(bucket_name_master)
                        # blob_master = bucket_master.blob(blob_name_master) # Asserted
                        self.mock_gcs_client_instance.bucket(bucket_name_master)
                        self.mock_gcs_bucket_instance.blob(blob_name_master)

                        self.mock_gcs_blob_instance.upload_from_string(
                            json.dumps(master_restaurant_data, indent=4), 
                            content_type='application/json'
                        )
                        self.mock_st_success(f"Successfully uploaded master restaurant data to {st_app.gcs_master_dictionary_output_uri}") # Updated success message
                except Exception as e:
                    self.mock_st_error(f"Error uploading master restaurant data to GCS: {e}") # Updated error message

        # DataFrame Creation part
        try:
            if not master_restaurant_data:
                self.mock_st_warning("No restaurant data to display (master restaurant data is empty after processing).") # Updated warning
            else:
                valid_items_for_df = [item for item in master_restaurant_data if isinstance(item, dict)]
                if not valid_items_for_df:
                    self.mock_st_warning("Master restaurant data contains no dictionary items, cannot display as table.") # Updated warning
                elif len(valid_items_for_df) < len(master_restaurant_data):
                    self.mock_st_warning(f"Some items in the master restaurant data were not dictionaries and were excluded from the table display. Displaying {len(valid_items_for_df)} records.") # Updated warning
                    self.mock_pd_normalize(valid_items_for_df) # Mocked pd.json_normalize
                else:
                    self.mock_pd_normalize(master_restaurant_data) # Mocked pd.json_normalize
        except Exception as e: 
            self.mock_st_error(f"Error displaying DataFrame from master restaurant data: {e}") # Updated error
        
        return master_restaurant_data, api_json_content
        # --- End of replicated logic ---


    def test_empty_master_restaurant_data_new_api_restaurants(self): # Renamed test
        self.mock_load_json_uri.return_value = None # Empty master restaurant data
        # Ensure GCS is not triggered for this specific test if not relevant
        st_app.gcs_destination_uri = "" # Override default from setUp if needed, or pass to helper
        
        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"FHRSID": 101, "BusinessName": "Restaurant A"},
                {"FHRSID": 102, "BusinessName": "Restaurant B"}
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        # To trigger the logic, we need to simulate the app's execution path.
        # The core logic is inside the 'if response.status_code == 200:' block in st_app.py
        # For this test, we assume st_app.master_list_uri is set, st_app.gcs_destination_uri can be empty
        # And then we need to simulate the data processing part
        
        # Simulate the state after API call and master list load
        st_app.data = api_response_data # Simulate data loaded from API
        
        # The actual merge logic in st_app.py happens after data is loaded and master_restaurant_data is attempted
        # We will call a helper that encapsulates this logic, or directly test its effects
        

        final_data = self._run_fetch_data_logic() # Explicitly no GCS for this test as gcs_destination_uri is "" by default in helper via setup

        self.assertEqual(len(final_data), 2)
        self.assertTrue(all('first_seen' in item and item['first_seen'] == self.fixed_date_str for item in final_data))
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(len(call_arg), 2)
        self.assertEqual(call_arg[0]['FHRSID'], 101)
        self.assertEqual(call_arg[1]['FHRSID'], 102)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called() # Ensure GCS was not called


    def test_populated_master_data_some_new_some_duplicates(self): # Renamed test
        master_data_input = [ # Renamed variable for clarity
            {"FHRSID": 101, "BusinessName": "Restaurant A", "first_seen": "2023-12-01"},
            {"FHRSID": 103, "BusinessName": "Restaurant C Old"} # No first_seen
        ]
        self.mock_load_json_uri.return_value = master_data_input

        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"FHRSID": 101, "BusinessName": "Restaurant A Updated in API"}, # Duplicate
                {"FHRSID": 102, "BusinessName": "Restaurant B New"},      # New
                {"FHRSID": 104, "BusinessName": "Restaurant D New"}       # New
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response
        
        st_app.data = api_response_data # Simulate

        final_data = self._run_fetch_data_logic() # Explicitly no GCS

        self.assertEqual(len(final_data), 4)
        self.mock_pd_normalize.assert_called_once()
        # ... (rest of assertions for this test remain the same)
        ids_in_final_data = {item['FHRSID'] for item in final_data} # Renamed var
        self.assertEqual(ids_in_final_data, {101, 102, 103, 104})
        item101 = next(item for item in final_data if item['FHRSID'] == 101)
        self.assertEqual(item101.get('first_seen'), "2023-12-01")
        self.assertEqual(item101.get('BusinessName'), "Restaurant A") # Name from master_data, not API
        item102 = next(item for item in final_data if item['FHRSID'] == 102)
        self.assertEqual(item102.get('first_seen'), self.fixed_date_str)
        item103 = next(item for item in final_data if item['FHRSID'] == 103)
        self.assertNotIn('first_seen', item103) # Original item without first_seen
        item104 = next(item for item in final_data if item['FHRSID'] == 104)
        self.assertEqual(item104.get('first_seen'), self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_empty_api_response_populated_master_data(self): # Renamed test
        master_data_input = [ # Renamed variable
            {"FHRSID": 201, "BusinessName": "Old Place"},
            {"FHRSID": 202, "BusinessName": "Another Old Place", "first_seen": "2023-01-01"}
        ]
        self.mock_load_json_uri.return_value = master_data_input

        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": []}}} # Empty API
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data

        final_data = self._run_fetch_data_logic()

        self.assertEqual(len(final_data), 2)
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(call_arg, master_data_input) 
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_master_data_loaded_as_dictionary_not_list(self): # Renamed to reflect what's tested
        # This test checks st_app's handling when load_json_from_uri returns a dict, not a list for master data.
        # The st_app logic should then proceed with an empty master_restaurant_data.
        master_data_as_dict = { # This is not a list, so st_app should treat it as invalid master data structure
            "FHRSID": 301, "BusinessName": "Master Dict Rest"
        }
        self.mock_load_json_uri.return_value = master_data_as_dict
        
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
             {"FHRSID": 302, "BusinessName": "API New Rest"} # API adds a new item
        ]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_data = self._run_fetch_data_logic()

        # Expected: master_restaurant_data should be empty (due to invalid input type)
        # and only the API data should be present.
        self.assertEqual(len(final_data), 1) 
        ids_in_final_data = {item['FHRSID'] for item in final_data}
        self.assertEqual(ids_in_final_data, {302})
        item302 = next(item for item in final_data if item['FHRSID'] == 302)
        self.assertEqual(item302.get('first_seen'), self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()
        # Check for the specific warning
        self.mock_st_warning.assert_any_call(f"Data loaded from {st_app.master_list_uri} is not in the expected format (e.g., a list of records). Type found: {type(master_data_as_dict)}. Proceeding with empty master restaurant data.")

    def test_master_data_loaded_as_list(self): # Renamed test
        master_data_as_list = [{"FHRSID": 401, "BusinessName": "Master List Rest"}] # Correct list format
        self.mock_load_json_uri.return_value = master_data_as_list
        
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
             {"FHRSID": 402, "BusinessName": "API New Rest From List Master"}
        ]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_data = self._run_fetch_data_logic()

        self.assertEqual(len(final_data), 2)
        ids_in_final_data = {item['FHRSID'] for item in final_data}
        self.assertEqual(ids_in_final_data, {401, 402})
        item402 = next(item for item in final_data if item['FHRSID'] == 402)
        self.assertEqual(item402.get('first_seen'), self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_fhrsid_missing_in_api_data_record(self): # Renamed
        self.mock_load_json_uri.return_value = [] # Empty master restaurant data
        
        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"BusinessName": "Restaurant NoID"}, 
                {"FHRSID": 501, "BusinessName": "Restaurant WithID"}
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_data = self._run_fetch_data_logic()
        
        self.assertEqual(len(final_data), 1)
        self.assertEqual(final_data[0]['FHRSID'], 501)
        self.assertEqual(final_data[0]['first_seen'], self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_fhrsid_missing_in_master_data_record(self): # Renamed
        master_data_input = [ # Renamed var
            {"BusinessName": "Master NoID"}, # This record is kept as is
            {"FHRSID": 601, "BusinessName": "Master WithID"}
        ]
        self.mock_load_json_uri.return_value = master_data_input
        
        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"FHRSID": 601, "BusinessName": "Master WithID From API"}, 
                {"FHRSID": 602, "BusinessName": "New API Rest"}
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_data = self._run_fetch_data_logic()
        
        self.assertEqual(len(final_data), 3) 
        ids_in_final_data_with_id = {item['FHRSID'] for item in final_data if 'FHRSID' in item}
        self.assertEqual(ids_in_final_data_with_id, {601, 602})
        item_master_noid = next(item for item in final_data if 'FHRSID' not in item)
        self.assertEqual(item_master_noid['BusinessName'], "Master NoID")
        item602 = next(item for item in final_data if item.get('FHRSID') == 602)
        self.assertEqual(item602['first_seen'], self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    # --- Tests for master restaurant data loading logic ---

    def _setup_minimal_api_response(self):
        """Helper to set up a minimal, valid API response for running fetch logic."""
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": []}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response
        st_app.data = api_response_data # Simulate data loaded from API for _run_fetch_data_logic

    def test_load_master_data_empty_list_from_uri(self): # Renamed
        # This test checks behavior when an empty list is loaded for master data.
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        self.mock_st_success.reset_mock()
        st_app.master_list_uri = "gs://dummy/empty_list.json" # URI must be non-empty
        self.mock_load_json_uri.return_value = [] # Mock load_json_from_uri to return an empty list
        self._setup_minimal_api_response() # Setup minimal API response to allow main logic to run

        result_data = self._run_fetch_data_logic()

        self.assertEqual(result_data, []) # Master restaurant data should be empty
        # Verify st.warning was called with the specific message for an empty list
        self.mock_st_warning.assert_any_call(f"Master restaurant data loaded from {st_app.master_list_uri}, but it's empty.")
        self.mock_st_success.assert_not_called() # No success message for loading data
        self.mock_st_info.assert_not_called() # No "No master restaurant data URI" message

    def test_load_master_data_is_none_from_uri(self): # Renamed
        # This test checks behavior when load_json_from_uri returns None for master data.
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        self.mock_st_success.reset_mock()
        st_app.master_list_uri = "gs://dummy/none_list.json" # URI must be non-empty
        self.mock_load_json_uri.return_value = None # Mock load_json_from_uri to return None
        self._setup_minimal_api_response()

        result_data = self._run_fetch_data_logic()

        self.assertEqual(result_data, []) # Master restaurant data should be empty
        # Verify st.warning was called with the specific message for load failure
        self.mock_st_warning.assert_any_call(f"Failed to load master restaurant data from {st_app.master_list_uri} (or it was empty/invalid). Proceeding with empty master restaurant data.")
        self.mock_st_success.assert_not_called()
        self.mock_st_info.assert_not_called()

    def test_no_master_restaurant_data_uri_provided(self): # Renamed
        # Checks behavior when no master restaurant data URI is provided.
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        self.mock_st_success.reset_mock()
        # Crucially, master_list_uri is effectively "" or None.
        # We achieve this by passing current_master_list_uri="" to the helper.
        self._setup_minimal_api_response()

        result_data = self._run_fetch_data_logic(current_master_list_uri="") # Simulate no URI

        self.assertEqual(result_data, []) # Master restaurant data should be empty
        self.mock_st_info.assert_any_call("No master restaurant data URI provided. Starting with empty master restaurant data.")
        self.mock_st_warning.assert_not_called()
        self.mock_st_success.assert_not_called() # No success for loading

    def test_load_master_data_is_string_from_uri(self): # Renamed
        # Checks behavior when loaded master data is a string (not a list).
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        self.mock_st_success.reset_mock()
        st_app.master_list_uri = "gs://dummy/string_list.json" # URI must be non-empty
        test_string = "this is not a list"
        self.mock_load_json_uri.return_value = test_string # Mock load_json_from_uri to return a string
        self._setup_minimal_api_response()

        result_data = self._run_fetch_data_logic()

        self.assertEqual(result_data, []) # Master restaurant data should be empty
        # Verify st.warning was called, indicating data was not a list
        self.mock_st_warning.assert_any_call(f"Data loaded from {st_app.master_list_uri} is not in the expected format (e.g., a list of records). Type found: {type(test_string)}. Proceeding with empty master restaurant data.")
        self.mock_st_success.assert_not_called()
        self.mock_st_info.assert_not_called()

    # --- Tests for various master data loading scenarios ---

    def test_load_master_data_valid_non_empty_list_from_uri(self): # Renamed
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        self.mock_st_success.reset_mock()
        st_app.master_list_uri = "gs://dummy/valid_list.json"
        valid_list_data = [{"id": 1, "name": "Restaurant Alpha"}]
        self.mock_load_json_uri.return_value = valid_list_data
        self._setup_minimal_api_response()

        result_data = self._run_fetch_data_logic()

        self.assertEqual(result_data, valid_list_data)
        self.mock_st_success.assert_any_call(f"Successfully loaded master restaurant data with {len(valid_list_data)} records from {st_app.master_list_uri}.")
        self.mock_st_warning.assert_not_called()
        self.mock_st_info.assert_not_called()

    def test_load_master_data_returns_dictionary_instead_of_list(self): # Renamed
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        self.mock_st_success.reset_mock()
        st_app.master_list_uri = "gs://dummy/dict_data.json"
        dict_data = {"key": "value", "another_key": "another_value"} # This is a dictionary, not a list
        self.mock_load_json_uri.return_value = dict_data
        self._setup_minimal_api_response()

        result_data = self._run_fetch_data_logic()

        self.assertEqual(result_data, []) # Master restaurant data should be empty
        self.mock_st_warning.assert_any_call(f"Data loaded from {st_app.master_list_uri} is not in the expected format (e.g., a list of records). Type found: {type(dict_data)}. Proceeding with empty master restaurant data.")
        self.mock_st_success.assert_not_called() # No success for loading
        self.mock_st_info.assert_not_called()

    # --- GCS Upload Tests ---

    def test_gcs_upload_api_response_only(self):
        st_app.gcs_destination_uri = "gs://api-bucket/api_folder/"
        st_app.gcs_master_dictionary_output_uri = "" # No master data upload

        test_api_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [{"FHRSID": 801, "BusinessName": "API Only Test"}]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = test_api_data
        self.mock_requests_get.return_value = mock_api_response
        
        self.mock_load_json_uri.return_value = [] # Empty master list initially

        # Reset GCS mocks before running the logic
        self.mock_gcs_client_instance.reset_mock()
        self.mock_gcs_bucket_instance.reset_mock()
        self.mock_gcs_blob_instance.reset_mock()
        # Ensure blob returns the main mock_gcs_blob_instance
        self.mock_gcs_bucket_instance.blob.return_value = self.mock_gcs_blob_instance


        _master_restaurant_data, api_data_content = self._run_fetch_data_logic() # Renamed var

        self.assertEqual(api_data_content, test_api_data)
        
        expected_api_filename = f"api_folder/api_response_{self.fixed_date_str}.json"
        self.mock_gcs_client_instance.bucket.assert_called_once_with("api-bucket")
        self.mock_gcs_bucket_instance.blob.assert_called_once_with(expected_api_filename)
        self.mock_gcs_blob_instance.upload_from_string.assert_called_once_with(
            json.dumps(test_api_data, indent=4),
            content_type='application/json'
        )
        self.mock_st_success.assert_any_call(f"Successfully uploaded raw API response to gs://api-bucket/{expected_api_filename}")

    def test_gcs_upload_master_restaurant_data_only(self): # Renamed test
        st_app.gcs_destination_uri = "" # No API response upload
        st_app.gcs_master_dictionary_output_uri = "gs://master-bucket/output/master_data.json" # Patched var, updated path for clarity

        initial_master_data = [{"FHRSID": 901, "BusinessName": "Master Only Original"}] # Renamed var
        self.mock_load_json_uri.return_value = initial_master_data
        
        # API returns one new item to make the final master data different
        test_api_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [{"FHRSID": 902, "BusinessName": "Master Only New API"}]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = test_api_data
        self.mock_requests_get.return_value = mock_api_response

        # Reset GCS mocks
        self.mock_gcs_client_instance.reset_mock()
        self.mock_gcs_bucket_instance.reset_mock()
        self.mock_gcs_blob_instance.reset_mock()
        self.mock_gcs_bucket_instance.blob.return_value = self.mock_gcs_blob_instance

        final_master_data, _api_data_content = self._run_fetch_data_logic() # Renamed var

        self.assertEqual(len(final_master_data), 2) # Original + new from API
        
        expected_master_blob_name = "output/master_data.json"
        self.mock_gcs_client_instance.bucket.assert_called_once_with("master-bucket")
        self.mock_gcs_bucket_instance.blob.assert_called_once_with(expected_master_blob_name)
        self.mock_gcs_blob_instance.upload_from_string.assert_called_once_with(
            json.dumps(final_master_data, indent=4),
            content_type='application/json'
        )
        self.mock_st_success.assert_any_call(f"Successfully uploaded master restaurant data to {st_app.gcs_master_dictionary_output_uri}") # Updated message

    def test_gcs_upload_both_api_and_master_data(self): # Renamed test
        st_app.gcs_destination_uri = "gs://dual-api-bucket/responses/"
        st_app.gcs_master_dictionary_output_uri = "gs://dual-master-bucket/data/final_master_data.json" # Patched var, updated path

        test_api_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [{"FHRSID": 1001, "BusinessName": "Dual Test API"}]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = test_api_data
        self.mock_requests_get.return_value = mock_api_response
        
        self.mock_load_json_uri.return_value = [] # Start with empty master

        # Reset GCS mocks
        self.mock_gcs_client_instance.reset_mock()
        self.mock_gcs_bucket_instance.reset_mock() # This will be the parent for blob mocks
        self.mock_gcs_blob_instance.reset_mock() # General blob instance, not used with side_effect

        # Setup side_effect for blob creation
        mock_api_blob_instance = MagicMock(name="APIBlob")
        mock_master_blob_instance = MagicMock(name="MasterBlob")
        
        expected_api_filename = f"responses/api_response_{self.fixed_date_str}.json"
        expected_master_filename = "dictionaries/final_master.json"

        def blob_side_effect(path):
            if path == expected_api_filename:
                return mock_api_blob_instance
            elif path == expected_master_filename:
                return mock_master_blob_instance
            self.fail(f"Unexpected blob path: {path}") # Fail test if path is unexpected
            return MagicMock() 
        self.mock_gcs_bucket_instance.blob.side_effect = blob_side_effect
        
        final_master_data, api_data_content = self._run_fetch_data_logic() # Renamed var

        # Assertions for API Response Upload
        mock_api_blob_instance.upload_from_string.assert_called_once_with(
            json.dumps(api_data_content, indent=4),
            content_type='application/json'
        )
        # Assertions for Master Restaurant Data Upload
        mock_master_blob_instance.upload_from_string.assert_called_once_with(
            json.dumps(final_master_data, indent=4), # Renamed var
            content_type='application/json'
        )
        # Check bucket calls (might be tricky with side_effect, ensure client.bucket was called)
        self.mock_gcs_client_instance.bucket.assert_any_call("dual-api-bucket")
        self.mock_gcs_client_instance.bucket.assert_any_call("dual-master-bucket")
        self.assertEqual(self.mock_gcs_client_instance.bucket.call_count, 2) # Called for each upload part
        
        # Check st.success messages
        self.mock_st_success.assert_any_call(f"Successfully uploaded raw API response to gs://dual-api-bucket/{expected_api_filename}")
        self.mock_st_success.assert_any_call(f"Successfully uploaded master restaurant data to {st_app.gcs_master_dictionary_output_uri}") # Updated message


    def test_gcs_upload_api_response_filename_construction(self):
        test_cases = [
            ("gs://filename-test/folder", f"folder/api_response_{self.fixed_date_str}.json", "filename-test"),
            ("gs://filename-test/folder/", f"folder/api_response_{self.fixed_date_str}.json", "filename-test"),
            ("gs://filename-test", f"api_response_{self.fixed_date_str}.json", "filename-test") 
        ]
        test_api_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": []}}} # Minimal API data
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = test_api_data
        self.mock_requests_get.return_value = mock_api_response
        self.mock_load_json_uri.return_value = [] # Empty master data
        st_app.gcs_master_dictionary_output_uri = "" # No master data upload

        for gcs_uri, expected_blob_name, expected_bucket_name in test_cases:
            st_app.gcs_destination_uri = gcs_uri
            
            self.mock_gcs_client_instance.reset_mock()
            self.mock_gcs_bucket_instance.reset_mock()
            self.mock_gcs_blob_instance.reset_mock()
            self.mock_gcs_bucket_instance.blob.return_value = self.mock_gcs_blob_instance
            self.mock_gcs_bucket_instance.blob.side_effect = None # Clear side effect if any
            self.mock_st_success.reset_mock() # Reset success calls for each case

            self._run_fetch_data_logic()

            self.mock_gcs_client_instance.bucket.assert_called_with(expected_bucket_name)
            self.mock_gcs_bucket_instance.blob.assert_called_with(expected_blob_name)
            self.mock_gcs_blob_instance.upload_from_string.assert_called_once()
            # Check success message for this specific call
            self.mock_st_success.assert_any_call(f"Successfully uploaded raw API response to gs://{expected_bucket_name}/{expected_blob_name}")


    def test_gcs_upload_master_data_error_missing_filename(self): # Renamed test
        st_app.gcs_destination_uri = "" # No API upload
        st_app.gcs_master_dictionary_output_uri = "gs://master-error-bucket/" # Invalid - missing filename for master data

        self.mock_load_json_uri.return_value = [{"FHRSID": 1101}] # Some master data
        self._setup_minimal_api_response() # Minimal API response

        self.mock_gcs_client_instance.reset_mock()
        self.mock_gcs_bucket_instance.reset_mock()
        self.mock_gcs_blob_instance.reset_mock()
        self.mock_st_error.reset_mock()

        self._run_fetch_data_logic()

        self.mock_st_error.assert_called_once_with("Invalid GCS URI for Master Restaurant Data: File name is missing.") # Updated error message
        # Ensure no upload attempt was made for the master data
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()
        # Ensure GCS client and bucket were called for the master data part, but not blob if error is caught early
        self.mock_gcs_client_instance.bucket.assert_called_once_with("master-error-bucket")
        # bucket.blob() for master data should not be called if filename is missing due to the check in st_app.
        self.mock_gcs_bucket_instance.blob.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
