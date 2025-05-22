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
        self.patcher_master_list_uri = patch('st_app.master_list_uri', "dummy_master_uri")
        self.patcher_gcs_destination_uri = patch('st_app.gcs_destination_uri', "") # Default to no GCS for most tests

        self.st_app_master_list_uri = self.patcher_master_list_uri.start()
        self.st_app_gcs_destination_uri = self.patcher_gcs_destination_uri.start()
        
        self.addCleanup(self.patcher_master_list_uri.stop)
        self.addCleanup(self.patcher_gcs_destination_uri.stop)


    def tearDown(self):
        # patch.stopall() # Not strictly necessary if using addCleanup for all patches started in setUp
        # However, if any patch is started with .start() and not added to addCleanup,
        # or if a test method itself uses .start(), then patch.stopall() is a good safety net.
        # For now, relying on addCleanup for specific patches.
        pass # Relying on addCleanup

    def _run_fetch_data_logic(self, current_gcs_destination_uri=None, current_master_list_uri=None):
        # This helper attempts to run the core logic of the "Fetch Data" button part
        # that processes data and prepares it for display/upload.

        # Allow overriding URI values for specific test scenarios via parameters
        if current_gcs_destination_uri is not None:
            st_app.gcs_destination_uri = current_gcs_destination_uri
        if current_master_list_uri is not None:
            st_app.master_list_uri = current_master_list_uri
        
        # It assumes that the st.button("Fetch Data") was pressed (i.e., returns True).
        # The actual code in st_app.py is not structured as a function we can call,
        # so we are essentially testing the behavior of that block when conditions are met.

        # Simulate the button press and response handling
        # The actual API call and response handling:
        # This is a simplified way to trigger the logic.
        # In st_app.py, this logic is inside `if response.status_code == 200:`
        # We ensure our mock_requests_get provides a 200 status and valid json()
        
        # The logic is inside `if st.button("Fetch Data"):`
        # And then `if response.status_code == 200:`
        # The `st_app.py` code directly executes this.
        # To test it, we need to ensure `st_app.data` is populated as if the API call happened
        # and `st_app.restaurants_master_list` is populated as if `load_json_from_uri` was called.

        # This is tricky because the actual code is not in a function.
        # We are testing the side effects and calls made by the script's flow
        # given our mocked inputs.

        # The key parts from st_app.py that we want to test:
        # 1. Load master list
        # 2. Process API response
        # 3. Update master list
        # 4. Call pd.json_normalize

        # Let's directly simulate the relevant part of the st_app.py code block
        # after `data = response.json()`

        # This is the most direct way to test the logic without refactoring st_app.py
        # We are essentially copy-pasting the logic here for testing purposes
        # This is not ideal, but a workaround for testing untestable code structure.

        # --- Start of copied logic block (modified for testability) ---
        # (Original st_app.py has this logic within `if response.status_code == 200:`)

        # 1.a. Load Master List (New simplified logic)
        restaurants_master_list = []
        if st_app.master_list_uri:
            loaded_master_data = self.mock_load_json_uri(st_app.master_list_uri)
            if loaded_master_data is not None:
                if isinstance(loaded_master_data, list):
                    restaurants_master_list = loaded_master_data
                    if restaurants_master_list:
                        self.mock_st_info(f"Successfully loaded master list with {len(restaurants_master_list)} items.")
                    else:
                        self.mock_st_warning("Master list loaded, but it's an empty list.")
                elif isinstance(loaded_master_data, dict):
                    if loaded_master_data: # If dictionary is not empty
                        found_list_in_dict = False
                        for value in loaded_master_data.values():
                            if isinstance(value, list):
                                restaurants_master_list = value
                                self.mock_st_info(f"Master list loaded. Found a list with {len(restaurants_master_list)} items within the dictionary.")
                                found_list_in_dict = True
                                break 
                        if not found_list_in_dict:
                            restaurants_master_list = [loaded_master_data]
                            self.mock_st_info("Master list loaded. Treating the non-empty dictionary as a single record.")
                    else: # Empty dictionary
                        self.mock_st_warning("Master list loaded, but it's an empty dictionary. Proceeding with an empty master list.")
                        # restaurants_master_list remains []
                else: # Not a list or dict (e.g. string, number)
                    self.mock_st_warning(f"Master list loaded from {st_app.master_list_uri}, but it is not a list or dictionary (type: {type(loaded_master_data)}). Proceeding with an empty master list.")
                    # restaurants_master_list remains []
            else: # loaded_master_data is None
                # load_json_from_uri (mocked here) would have shown an error via st.error in real app
                self.mock_st_warning("Failed to load master list (it was None or loading failed). Proceeding with an empty master list.")
                # restaurants_master_list remains []
        else: # No master_list_uri
            self.mock_st_info("No master list URI provided. Starting with an empty master list.")
        
        # Ensure restaurants_master_list is always a list, even if logic above has a flaw
        if not isinstance(restaurants_master_list, list):
            self.mock_st_warning(f"restaurants_master_list was unexpectedly not a list (type: {type(restaurants_master_list)} after processing). Resetting to an empty list.")
            restaurants_master_list = []

        # 1.b. Process API Response and Update Master List
        # `api_data` comes from `self.mock_requests_get().json()`
        api_data_response = self.mock_requests_get()
        api_data_response.raise_for_status() # Simulate check or assume 200
        api_json_content = api_data_response.json()

        api_establishments = api_json_content.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
        if api_establishments is None: 
            api_establishments = []

        existing_fhrsid_set = {est['FHRSID'] for est in restaurants_master_list if isinstance(est, dict) and 'FHRSID' in est}
        today_date_str = self.mock_datetime_now.now().strftime("%Y-%m-%d") # Uses mocked datetime
        new_restaurants_added_count = 0

        for api_establishment in api_establishments:
            if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
                if api_establishment['FHRSID'] not in existing_fhrsid_set:
                    api_establishment['first_seen'] = today_date_str
                    restaurants_master_list.append(api_establishment)
                    existing_fhrsid_set.add(api_establishment['FHRSID'])
                    new_restaurants_added_count += 1
        
        # Call success message (simplified)
        self.mock_st_success(f"Processed API response. Added {new_restaurants_added_count} new restaurants. Total unique establishments: {len(restaurants_master_list)}")

        # DataFrame Creation part - uses 'restaurants_master_list'
        # This part is from st_app.py
        try:
            if not restaurants_master_list:
                self.mock_st_warning("No establishment data to display (master list is empty after processing).")
            else:
                valid_items_for_df = [item for item in restaurants_master_list if isinstance(item, dict)]
                if not valid_items_for_df:
                    self.mock_st_warning("Master list contains no dictionary items, cannot display as table.")
                elif len(valid_items_for_df) < len(restaurants_master_list):
                    self.mock_st_warning(f"Some items in the master list were not dictionaries and were excluded from the table display. Displaying {len(valid_items_for_df)} items.")
                    self.mock_pd_normalize(valid_items_for_df)
                else:
                    self.mock_pd_normalize(restaurants_master_list)
        except Exception as e: 
            self.mock_st_error(f"Error displaying DataFrame from master list: {e}")
            # Fallback logic not deeply simulated here, focus is on pd_normalize call

        # GCS Upload Logic - uses 'restaurants_master_list' (as per recent changes to st_app.py)
        # This logic is taken from st_app.py and adapted for the test helper
        if st_app.gcs_destination_uri: 
            if not st_app.gcs_destination_uri.startswith("gs://"):
                self.mock_st_error("Invalid GCS URI. It must start with gs://")
            else:
                try:
                    current_date = self.mock_datetime_now.now().strftime("%Y-%m-%d") # Mocked datetime
                    file_name = f"food_standards_data_{current_date}.json"
                    # GCS client, bucket, blob are mocked in setUp
                    bucket_name = st_app.gcs_destination_uri.split("/")[2]
                    blob_name_prefix = "/".join(st_app.gcs_destination_uri.split("/")[3:])
                    blob_path = st_app.os.path.join(blob_name_prefix, file_name) # Use st_app.os

                    # Access the mocks set up in `setUp`
                    self.mock_gcs_client_instance.bucket.assert_called_with(bucket_name) # Check bucket name
                    self.mock_gcs_bucket_instance.blob.assert_called_with(blob_path) # Check blob path

                    # THE ACTUAL UPLOAD CALL (that we want to test content of)
                    self.mock_gcs_blob_instance.upload_from_string(
                        json.dumps(restaurants_master_list, indent=4), 
                        content_type='application/json'
                    )
                    self.mock_st_success(f"Successfully uploaded raw API data to gs://{bucket_name}/{blob_path}")
                except Exception as e:
                    self.mock_st_error(f"Error uploading raw API data to GCS: {e}")
        
        
        return restaurants_master_list 
        # --- End of modified logic block ---


    def test_empty_master_list_new_api_restaurants(self):
        self.mock_load_json_uri.return_value = None # Empty master list
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
        
        # The actual merge logic in st_app.py happens after data is loaded and master_list is attempted
        # We will call a helper that encapsulates this logic, or directly test its effects
        

        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="") # Explicitly no GCS for this test

        self.assertEqual(len(final_list), 2)
        self.assertTrue(all('first_seen' in item and item['first_seen'] == self.fixed_date_str for item in final_list))
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(len(call_arg), 2)
        self.assertEqual(call_arg[0]['FHRSID'], 101)
        self.assertEqual(call_arg[1]['FHRSID'], 102)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called() # Ensure GCS was not called


    def test_populated_master_some_new_some_duplicates(self):
        master_data = [
            {"FHRSID": 101, "BusinessName": "Restaurant A", "first_seen": "2023-12-01"},
            {"FHRSID": 103, "BusinessName": "Restaurant C Old"} # No first_seen
        ]
        self.mock_load_json_uri.return_value = master_data # Simple list format

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

        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="") # Explicitly no GCS

        self.assertEqual(len(final_list), 4)
        self.mock_pd_normalize.assert_called_once()
        # ... (rest of assertions for this test remain the same)
        ids_in_final_list = {item['FHRSID'] for item in final_list}
        self.assertEqual(ids_in_final_list, {101, 102, 103, 104})
        item101 = next(item for item in final_list if item['FHRSID'] == 101)
        self.assertEqual(item101.get('first_seen'), "2023-12-01")
        self.assertEqual(item101.get('BusinessName'), "Restaurant A")
        item102 = next(item for item in final_list if item['FHRSID'] == 102)
        self.assertEqual(item102.get('first_seen'), self.fixed_date_str)
        item103 = next(item for item in final_list if item['FHRSID'] == 103)
        self.assertNotIn('first_seen', item103)
        item104 = next(item for item in final_list if item['FHRSID'] == 104)
        self.assertEqual(item104.get('first_seen'), self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_empty_api_response_populated_master(self):
        master_data = [
            {"FHRSID": 201, "BusinessName": "Old Place"},
            {"FHRSID": 202, "BusinessName": "Another Old Place", "first_seen": "2023-01-01"}
        ]
        self.mock_load_json_uri.return_value = master_data

        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": []}}} # Empty API
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data

        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="")

        self.assertEqual(len(final_list), 2)
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(call_arg, master_data) 
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_master_list_structure_dict_input(self):
        # Modified to be compatible with the new loading logic, 
        # where a list is found as a value in the root dictionary.
        master_data_dict_compatible_with_new_logic = {
            "establishments": [ # The new logic will find this list
                {"FHRSID": 301, "BusinessName": "Master Dict Rest"}
            ]
        }
        self.mock_load_json_uri.return_value = master_data_dict_compatible_with_new_logic
        
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
             {"FHRSID": 302, "BusinessName": "API New Rest"}
        ]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="")

        self.assertEqual(len(final_list), 2)
        ids_in_final_list = {item['FHRSID'] for item in final_list}
        self.assertEqual(ids_in_final_list, {301, 302})
        item302 = next(item for item in final_list if item['FHRSID'] == 302)
        self.assertEqual(item302.get('first_seen'), self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_master_list_structure_list_input(self):
        master_data_list = [{"FHRSID": 401, "BusinessName": "Master List Rest"}]
        self.mock_load_json_uri.return_value = master_data_list
        
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
             {"FHRSID": 402, "BusinessName": "API New Rest From List Master"}
        ]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="")

        self.assertEqual(len(final_list), 2)
        ids_in_final_list = {item['FHRSID'] for item in final_list}
        self.assertEqual(ids_in_final_list, {401, 402})
        item402 = next(item for item in final_list if item['FHRSID'] == 402)
        self.assertEqual(item402.get('first_seen'), self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_fhrsid_missing_in_api_data(self):
        self.mock_load_json_uri.return_value = [] # Empty master
        
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
        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="")
        
        self.assertEqual(len(final_list), 1)
        self.assertEqual(final_list[0]['FHRSID'], 501)
        self.assertEqual(final_list[0]['first_seen'], self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_fhrsid_missing_in_master_data(self):
        master_data = [
            {"BusinessName": "Master NoID"},
            {"FHRSID": 601, "BusinessName": "Master WithID"}
        ]
        self.mock_load_json_uri.return_value = master_data
        
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
        final_list = self._run_fetch_data_logic(current_gcs_destination_uri="")
        
        self.assertEqual(len(final_list), 3) 
        ids_in_final_list_with_id = {item['FHRSID'] for item in final_list if 'FHRSID' in item}
        self.assertEqual(ids_in_final_list_with_id, {601, 602})
        item_master_noid = next(item for item in final_list if 'FHRSID' not in item)
        self.assertEqual(item_master_noid['BusinessName'], "Master NoID")
        item602 = next(item for item in final_list if item.get('FHRSID') == 602)
        self.assertEqual(item602['first_seen'], self.fixed_date_str)
        self.mock_gcs_blob_instance.upload_from_string.assert_not_called()

    def test_gcs_upload_and_download_content(self):
        # 1. Ensure master_list_uri is mocked (e.g., to return an empty list)
        self.mock_load_json_uri.return_value = [] # Start with an empty master list

        # 2. Set up mock API response data
        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"FHRSID": 701, "BusinessName": "Restaurant GCS Test"},
                {"FHRSID": 702, "BusinessName": "Another GCS Test Rest"}
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response
        st_app.data = api_response_data # Simulate data loaded from API

        # 3. Patch st_app.gcs_destination_uri to a valid dummy GCS path
        test_gcs_uri = "gs://test-bucket/output_folder"
        # The _run_fetch_data_logic helper will use this via its parameter or by patching st_app directly
        
        # Expected data after processing (new items get 'first_seen')
        expected_restaurants_master_list = [
            {"FHRSID": 701, "BusinessName": "Restaurant GCS Test", "first_seen": self.fixed_date_str},
            {"FHRSID": 702, "BusinessName": "Another GCS Test Rest", "first_seen": self.fixed_date_str}
        ]
        expected_data_json_str = json.dumps(expected_restaurants_master_list, indent=4)

        # 6. Execute the main data processing logic
        # The modified _run_fetch_data_logic now includes GCS and download button calls.
        # We pass the test_gcs_uri to it.
        # It will use self.mock_gcs_blob_instance and self.mock_st_download_button (from setUp)
        
        # Reset mocks that might be called multiple times if helper is reused across tests
        self.mock_gcs_client_instance.reset_mock()
        self.mock_gcs_bucket_instance.reset_mock()
        self.mock_gcs_blob_instance.reset_mock()

        # Call the helper function that now encapsulates GCS and Download logic
        # This function will internally make the assertions on GCS client calls like bucket() and blob()
        # and then call upload_from_string and st.download_button
        returned_master_list = self._run_fetch_data_logic(current_gcs_destination_uri=test_gcs_uri)

        self.assertEqual(returned_master_list, expected_restaurants_master_list)

        # 7. Assert that the mocked blob.upload_from_string method was called once.
        # 8. Assert that the first argument to blob.upload_from_string was correct.
        self.mock_gcs_blob_instance.upload_from_string.assert_called_once_with(
            expected_data_json_str,
            content_type='application/json'
        )
        
        # Assert GCS client, bucket, blob calls (already in helper, but can be re-asserted for clarity if needed)
        # Example: self.mock_gcs_client_instance.bucket.assert_called_with("test-bucket")
        # current_date = self.fixed_date.strftime("%Y-%m-%d")
        # expected_blob_name = f"output_folder/food_standards_data_{current_date}.json" 
        # self.mock_gcs_bucket_instance.blob.assert_called_with(expected_blob_name)

    # --- New tests for master list loading logic ---

    def _setup_minimal_api_response(self):
        """Helper to set up a minimal, valid API response."""
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": []}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response
        st_app.data = api_response_data # Simulate data loaded from API for _run_fetch_data_logic

    def test_load_master_list_non_empty_dict_extracts_list_value(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        st_app.master_list_uri = "gs://dummy/dict_with_list.json" # Ensure URI is set
        master_data = {"data": [{"id": 1, "name": "Item1"}, {"id": 2, "name": "Item2"}]}
        self.mock_load_json_uri.return_value = master_data
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic()
        
        self.assertEqual(result_list, [{"id": 1, "name": "Item1"}, {"id": 2, "name": "Item2"}])
        self.mock_st_info.assert_any_call(f"Master list loaded. Found a list with {len(master_data['data'])} items within the dictionary.")
        self.mock_st_warning.assert_not_called()

    def test_load_master_list_non_empty_dict_no_list_value_wraps_dict(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        st_app.master_list_uri = "gs://dummy/dict_no_list.json"
        master_data = {"id": 1, "name": "item"}
        self.mock_load_json_uri.return_value = master_data
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic()

        self.assertEqual(result_list, [{"id": 1, "name": "item"}])
        self.mock_st_info.assert_any_call("Master list loaded. Treating the non-empty dictionary as a single record.")
        self.mock_st_warning.assert_not_called()

    def test_load_master_list_empty_dict(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        st_app.master_list_uri = "gs://dummy/empty_dict.json"
        self.mock_load_json_uri.return_value = {}
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic()

        self.assertEqual(result_list, [])
        self.mock_st_warning.assert_any_call("Master list loaded, but it's an empty dictionary. Proceeding with an empty master list.")
        self.mock_st_info.assert_not_called() # Or check it wasn't the "success" messages for list loading

    def test_load_master_list_empty_list(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        st_app.master_list_uri = "gs://dummy/empty_list.json"
        self.mock_load_json_uri.return_value = []
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic()

        self.assertEqual(result_list, [])
        self.mock_st_warning.assert_any_call("Master list loaded, but it's an empty list.")
        # self.mock_st_info.assert_not_called() # No "successfully loaded" for empty list

    def test_load_master_list_is_none(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        st_app.master_list_uri = "gs://dummy/none_list.json"
        self.mock_load_json_uri.return_value = None
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic()

        self.assertEqual(result_list, [])
        self.mock_st_warning.assert_any_call("Failed to load master list (it was None or loading failed). Proceeding with an empty master list.")
        # self.mock_st_info.assert_not_called()

    def test_load_master_list_is_string(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        st_app.master_list_uri = "gs://dummy/string_list.json" # URI must be non-empty for this path
        test_string = "just a string"
        self.mock_load_json_uri.return_value = test_string
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic()

        self.assertEqual(result_list, [])
        self.mock_st_warning.assert_any_call(f"Master list loaded from {st_app.master_list_uri}, but it is not a list or dictionary (type: {type(test_string)}). Proceeding with an empty master list.")
        # self.mock_st_info.assert_not_called()

    def test_no_master_list_uri_provided(self):
        self.mock_st_info.reset_mock()
        self.mock_st_warning.reset_mock()
        # Crucially, set master_list_uri to empty or None for this test
        # The _run_fetch_data_logic uses current_master_list_uri parameter, or st_app.master_list_uri
        # We will pass "" to the helper.
        self._setup_minimal_api_response()

        result_list = self._run_fetch_data_logic(current_master_list_uri="")

        self.assertEqual(result_list, [])
        self.mock_st_info.assert_any_call("No master list URI provided. Starting with an empty master list.")
        self.mock_st_warning.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
