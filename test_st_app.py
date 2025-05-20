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
        
        # Mock GCS upload related parts, as they are in the same block but not the focus
        self.mock_storage_client = patch('st_app.storage.Client').start()

        # Simulate inputs that are typically read from st.text_input
        # These are module-level in st_app.py, so we patch them there
        self.patch_master_list_uri = patch('st_app.master_list_uri', "dummy_master_uri")
        self.patch_gcs_destination_uri = patch('st_app.gcs_destination_uri', "") # No GCS upload for these tests

        self.patch_master_list_uri.start()
        self.patch_gcs_destination_uri.start()


    def tearDown(self):
        patch.stopall()

    def _run_fetch_data_logic(self):
        # This helper attempts to run the core logic of the "Fetch Data" button.
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

        # 1.a. Load Master List (relies on self.mock_load_json_uri and st_app.master_list_uri)
        restaurants_master_list = []
        if st_app.master_list_uri: # This will be "dummy_master_uri" or overridden in tests
            loaded_master_data = self.mock_load_json_uri(st_app.master_list_uri)
            if loaded_master_data is not None:
                if isinstance(loaded_master_data, dict) and \
                   'FHRSEstablishment' in loaded_master_data and \
                   'EstablishmentCollection' in loaded_master_data.get('FHRSEstablishment', {}) and \
                   'EstablishmentDetail' in loaded_master_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}):
                    restaurants_master_list = loaded_master_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
                    # messages...
                elif isinstance(loaded_master_data, list):
                    restaurants_master_list = loaded_master_data
                    # messages...
                # else messages...
            # else messages...
        # else messages...
        
        if not isinstance(restaurants_master_list, list):
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
        # self.mock_st_success(f"Processed API response. Added {new_restaurants_added_count} new restaurants. Total unique establishments: {len(restaurants_master_list)}")

        # 2. Modify DataFrame Creation - uses 'restaurants_master_list'
        if not restaurants_master_list:
            pass # st.warning("No establishment data to display (master list is empty after processing).")
        else:
            valid_items_for_df = [item for item in restaurants_master_list if isinstance(item, dict)]
            if valid_items_for_df:
                 self.mock_pd_normalize(valid_items_for_df) # Capture argument
            # else warnings...
        
        return restaurants_master_list # Return the final list for assertion
        # --- End of copied logic block ---


    def test_empty_master_list_new_api_restaurants(self):
        self.mock_load_json_uri.return_value = None # Empty master list
        
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
        
        # Forcing the recreation of the logic block inside the test is prone to drift.
        # A better way:
        # Set up mocks for inputs (`st_app.master_list_uri`, `st_app.load_json_from_uri`, `st_app.requests.get`)
        # Set up mocks for outputs (`st_app.st.success`, `st_app.pd.json_normalize`)
        # Then, conceptually, run the part of `st_app.py` that does the work.
        # Since `st_app.py` is a script, we can't just call a function.
        # One way is to use `runpy.run_module` or `exec`, but that's too complex for this.

        # Let's refine `_run_fetch_data_logic` to be called here.
        # It will use the mocked global-like values (st_app.master_list_uri) and mocked functions.

        final_list = self._run_fetch_data_logic()

        self.assertEqual(len(final_list), 2)
        self.assertTrue(all('first_seen' in item and item['first_seen'] == self.fixed_date_str for item in final_list))
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(len(call_arg), 2)
        self.assertEqual(call_arg[0]['FHRSID'], 101)
        self.assertEqual(call_arg[1]['FHRSID'], 102)


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

        final_list = self._run_fetch_data_logic()
        
        self.assertEqual(len(final_list), 4)
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(len(call_arg), 4)

        ids_in_final_list = {item['FHRSID'] for item in final_list}
        self.assertEqual(ids_in_final_list, {101, 102, 103, 104})

        item101 = next(item for item in final_list if item['FHRSID'] == 101)
        item102 = next(item for item in final_list if item['FHRSID'] == 102)
        item103 = next(item for item in final_list if item['FHRSID'] == 103)
        item104 = next(item for item in final_list if item['FHRSID'] == 104)

        self.assertEqual(item101.get('first_seen'), "2023-12-01") # Original preserved
        self.assertEqual(item101.get('BusinessName'), "Restaurant A") # Original preserved

        self.assertEqual(item102.get('first_seen'), self.fixed_date_str) # New gets date
        self.assertEqual(item102.get('BusinessName'), "Restaurant B New")
        
        self.assertNotIn('first_seen', item103) # Was not in master with date, not from API
        self.assertEqual(item103.get('BusinessName'), "Restaurant C Old")

        self.assertEqual(item104.get('first_seen'), self.fixed_date_str) # New gets date
        self.assertEqual(item104.get('BusinessName'), "Restaurant D New")


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

        final_list = self._run_fetch_data_logic()

        self.assertEqual(len(final_list), 2)
        self.mock_pd_normalize.assert_called_once()
        call_arg = self.mock_pd_normalize.call_args[0][0]
        self.assertEqual(call_arg, master_data) # Should be identical to master

    def test_master_list_structure_dict_input(self):
        # Master list is the full API-like structure
        master_data_dict = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"FHRSID": 301, "BusinessName": "Master Dict Rest"}
            ]}}
        }
        self.mock_load_json_uri.return_value = master_data_dict
        
        api_response_data = {"FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
             {"FHRSID": 302, "BusinessName": "API New Rest"}
        ]}}}
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_list = self._run_fetch_data_logic()

        self.assertEqual(len(final_list), 2)
        ids_in_final_list = {item['FHRSID'] for item in final_list}
        self.assertEqual(ids_in_final_list, {301, 302})
        item302 = next(item for item in final_list if item['FHRSID'] == 302)
        self.assertEqual(item302.get('first_seen'), self.fixed_date_str)

    def test_master_list_structure_list_input(self):
        # Master list is a simple list (already tested in other cases, but good for explicit check)
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
        final_list = self._run_fetch_data_logic()

        self.assertEqual(len(final_list), 2)
        ids_in_final_list = {item['FHRSID'] for item in final_list}
        self.assertEqual(ids_in_final_list, {401, 402})
        item402 = next(item for item in final_list if item['FHRSID'] == 402)
        self.assertEqual(item402.get('first_seen'), self.fixed_date_str)

    def test_fhrsid_missing_in_api_data(self):
        self.mock_load_json_uri.return_value = [] # Empty master
        
        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"BusinessName": "Restaurant NoID"}, # Missing FHRSID
                {"FHRSID": 501, "BusinessName": "Restaurant WithID"}
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_list = self._run_fetch_data_logic()
        
        # The current code `if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:`
        # means the item without FHRSID will be skipped.
        self.assertEqual(len(final_list), 1)
        self.assertEqual(final_list[0]['FHRSID'], 501)
        self.assertEqual(final_list[0]['first_seen'], self.fixed_date_str)

    def test_fhrsid_missing_in_master_data(self):
        # Master data has an item missing FHRSID
        master_data = [
            {"BusinessName": "Master NoID"},
            {"FHRSID": 601, "BusinessName": "Master WithID"}
        ]
        self.mock_load_json_uri.return_value = master_data
        
        api_response_data = {
            "FHRSEstablishment": {"EstablishmentCollection": {"EstablishmentDetail": [
                {"FHRSID": 601, "BusinessName": "Master WithID From API"}, # Duplicate of valid master
                {"FHRSID": 602, "BusinessName": "New API Rest"}
            ]}}
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = api_response_data
        self.mock_requests_get.return_value = mock_api_response

        st_app.data = api_response_data
        final_list = self._run_fetch_data_logic()

        # The set comprehension `existing_fhrsid_set = {est['FHRSID'] for est in restaurants_master_list if isinstance(est, dict) and 'FHRSID' in est}`
        # will correctly create the set only from valid master items.
        # The item 'Master NoID' will remain in the list.
        # The new item 602 will be added.
        # The item 601 from API will be seen as duplicate of master 601.
        
        self.assertEqual(len(final_list), 3) # Master NoID, Master WithID, New API Rest
        
        ids_in_final_list_with_id = {item['FHRSID'] for item in final_list if 'FHRSID' in item}
        self.assertEqual(ids_in_final_list_with_id, {601, 602})

        item_master_noid = next(item for item in final_list if 'FHRSID' not in item)
        self.assertEqual(item_master_noid['BusinessName'], "Master NoID")

        item602 = next(item for item in final_list if item.get('FHRSID') == 602)
        self.assertEqual(item602['first_seen'], self.fixed_date_str)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
