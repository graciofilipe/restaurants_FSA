import unittest # Changed from pytest to unittest for consistency with TestAppendToBigQuery
from unittest.mock import MagicMock, patch
from data_processing import load_master_data, process_and_update_master_data, load_json_from_local_file_path # Added load_json_from_local_file_path
from bq_utils import ORIGINAL_COLUMNS_TO_KEEP # Import ORIGINAL_COLUMNS_TO_KEEP
from datetime import datetime
import pandas as pd # Added for potential pd.NA usage if needed by tested functions directly
import json # For load_json_from_local_file_path tests

# --- Tests for load_json_from_local_file_path ---
class TestLoadJsonFromLocalFilePath(unittest.TestCase):
    @patch('data_processing.open', new_callable=unittest.mock.mock_open, read_data='{"key": "value"}')
    @patch('data_processing.json.load')
    @patch('data_processing.st') # Mock streamlit
    def test_load_json_success(self, mock_st, mock_json_load, mock_file_open):
        mock_json_load.return_value = {"key": "value"}
        result = load_json_from_local_file_path("dummy_path.json")
        self.assertEqual(result, {"key": "value"})
        mock_file_open.assert_called_once_with("dummy_path.json", 'r')
        mock_json_load.assert_called_once()
        mock_st.error.assert_not_called()

    @patch('data_processing.open', side_effect=FileNotFoundError("File not found"))
    @patch('data_processing.st') # Mock streamlit
    def test_load_json_file_not_found(self, mock_st, mock_file_open):
        result = load_json_from_local_file_path("non_existent.json")
        self.assertIsNone(result)
        mock_st.error.assert_called_once_with("Error: Local file not found at non_existent.json")

    @patch('data_processing.open', new_callable=unittest.mock.mock_open, read_data='invalid json')
    @patch('data_processing.json.load', side_effect=json.JSONDecodeError("Error decoding", "doc", 0))
    @patch('data_processing.st') # Mock streamlit
    def test_load_json_decode_error(self, mock_st, mock_json_load, mock_file_open):
        result = load_json_from_local_file_path("invalid_format.json")
        self.assertIsNone(result)
        mock_st.error.assert_called_once() # Error message format can be checked more specifically if needed

    @patch('data_processing.open', side_effect=Exception("Some other error"))
    @patch('data_processing.st') # Mock streamlit
    def test_load_json_other_exception(self, mock_st, mock_file_open):
        result = load_json_from_local_file_path("other_error.json")
        self.assertIsNone(result)
        mock_st.error.assert_called_once_with("Error reading local file other_error.json: Some other error")


# --- Tests for load_master_data (modified) ---
class TestLoadMasterData(unittest.TestCase):
    @patch('data_processing.st')
    def test_load_master_data_success_and_manual_review_init(self, mock_st):
        # Mock for the load_bq_func argument
        mock_bq_loader = MagicMock(return_value=[
            {'FHRSID': "1", 'name': 'Restaurant A'}, # FHRSID is string
            {'FHRSID': "2", 'name': 'Restaurant B', 'manual_review': 'already_reviewed'} # FHRSID is string
        ])

        project_id = "test_p"
        dataset_id = "test_d"
        table_id = "test_t"

        result = load_master_data(project_id, dataset_id, table_id, mock_bq_loader)

        mock_bq_loader.assert_called_once_with(project_id, dataset_id, table_id)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['manual_review'], 'not reviewed') # Initialized
        self.assertEqual(result[1]['manual_review'], 'already_reviewed') # Preserved
        mock_st.success.assert_called_once() # Assuming success is logged

    @patch('data_processing.st')
    def test_load_master_data_empty_from_bq(self, mock_st):
        mock_bq_loader = MagicMock(return_value=[])
        result = load_master_data("p", "d", "t", mock_bq_loader)
        self.assertEqual(result, [])
        mock_st.info.assert_any_call("Master restaurant data loaded from BigQuery table p.d.t, but the table is empty or returned no data.")

    @patch('data_processing.st')
    def test_load_master_data_bq_func_returns_none(self, mock_st):
        mock_bq_loader = MagicMock(return_value=None) # Simulate BQ function returning None
        result = load_master_data("p", "d", "t", mock_bq_loader)
        self.assertEqual(result, [])
        mock_st.warning.assert_called_once_with("Failed to load master restaurant data from BigQuery table p.d.t (function returned None). Proceeding with empty master restaurant data.")

    @patch('data_processing.st')
    def test_load_master_data_bq_func_raises_exception(self, mock_st):
        mock_bq_loader = MagicMock(side_effect=Exception("BigQuery Load Error"))
        result = load_master_data("p", "d", "t", mock_bq_loader)
        self.assertEqual(result, [])
        mock_st.error.assert_called_once_with("An error occurred while calling load_bq_func for p.d.t: BigQuery Load Error")

    @patch('data_processing.st')
    def test_load_master_data_non_list_from_bq(self, mock_st):
        mock_bq_loader = MagicMock(return_value={"data": "not a list"}) # Simulate BQ function returning non-list
        result = load_master_data("p", "d", "t", mock_bq_loader)
        self.assertEqual(result, [])
        mock_st.error.assert_called_once_with("Data loaded from BigQuery table p.d.t is not in the expected list format. Type found: <class 'dict'>. Proceeding with empty master restaurant data.")


# --- Tests for process_and_update_master_data (modified) ---
class TestProcessAndUpdateMasterData(unittest.TestCase):
    @patch('data_processing.st') # Mock streamlit
    def test_no_new_restaurants(self, mock_st):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [{'FHRSID': "1", 'name': 'A'}]}}} # FHRSID is string
        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        mock_st.info.assert_called_once_with("Processed API response. No new restaurant records identified (or all were duplicates within the batch or already in BigQuery).")

    @patch('data_processing.datetime') # Mock datetime for predictable first_seen
    @patch('data_processing.st') # Mock streamlit
    def test_add_new_restaurants_and_fields_initialization(self, mock_st, mock_datetime):
        # Setup mock for datetime.now().strftime()
        mock_datetime_str = "2023-10-26"
        mock_datetime.now.return_value.strftime.return_value = mock_datetime_str

        master_data = [{'FHRSID': "1", 'BusinessName': 'A'}] # Existing record

        # Define API data with one existing and two new restaurants
        # These can have extra fields not in ORIGINAL_COLUMNS_TO_KEEP
        api_restaurant_1_existing = {'FHRSID': "1", 'BusinessName': 'A_updated', 'RatingValue': "Awful"}
        api_restaurant_2_new = {
            'FHRSID': "2", 'BusinessName': 'Cafe Terra', 'RatingValue': '5',
            'AddressLine1': '123 Main St', 'PostCode': 'AB1 2CD',
            'LocalAuthorityName': 'Test Council', 'NewRatingPending': 'false',
            'Scores': {'Hygiene': 10}, 'Geocode': {'Latitude': '1.0'}, 'BusinessType': 'Cafe'
        }
        api_restaurant_3_new = { # Minimal data, missing some optional ORIGINAL_COLUMNS_TO_KEEP fields
            'FHRSID': "3", 'BusinessName': 'Pizza Place', 'RatingValue': '4',
            'NewRatingPending': 'True', # String true
            # Missing AddressLine1, PostCode, LocalAuthorityName from ORIGINAL_COLUMNS_TO_KEEP
            'RatingDate': "2023-01-01" # This field is not in ORIGINAL_COLUMNS_TO_KEEP
        }

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            api_restaurant_1_existing,
            api_restaurant_2_new,
            api_restaurant_3_new
        ]}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 2)

        # Check properties of the new restaurants
        for r_new in new_restaurants:
            self.assertEqual(set(r_new.keys()), set(ORIGINAL_COLUMNS_TO_KEEP))
            self.assertEqual(r_new['first_seen'], mock_datetime_str)
            self.assertEqual(r_new['manual_review'], "not reviewed")
            self.assertIsNone(r_new.get('gemini_insights')) # Should be None as it's not in API mock

            if r_new['FHRSID'] == "2": # api_restaurant_2_new
                self.assertEqual(r_new['BusinessName'], 'Cafe Terra')
                self.assertEqual(r_new['RatingValue'], '5')
                self.assertEqual(r_new['AddressLine1'], '123 Main St')
                self.assertEqual(r_new['PostCode'], 'AB1 2CD')
                self.assertEqual(r_new['LocalAuthorityName'], 'Test Council')
                self.assertEqual(r_new['NewRatingPending'], 'false') # Kept as string from API
                # Optional fields from ORIGINAL_COLUMNS_TO_KEEP not in API mock for this item should be None
                self.assertIsNone(r_new.get('AddressLine2'))
                self.assertIsNone(r_new.get('AddressLine3'))
            elif r_new['FHRSID'] == "3": # api_restaurant_3_new
                self.assertEqual(r_new['BusinessName'], 'Pizza Place')
                self.assertEqual(r_new['RatingValue'], '4')
                self.assertEqual(r_new['NewRatingPending'], 'True') # Kept as string from API
                # These were missing in API data, so should be None
                self.assertIsNone(r_new.get('AddressLine1'))
                self.assertIsNone(r_new.get('AddressLine2'))
                self.assertIsNone(r_new.get('AddressLine3'))
                self.assertIsNone(r_new.get('PostCode'))
                self.assertIsNone(r_new.get('LocalAuthorityName'))

            # Assert that fields NOT in ORIGINAL_COLUMNS_TO_KEEP are absent
            self.assertNotIn('Scores', r_new)
            self.assertNotIn('Geocode', r_new)
            self.assertNotIn('BusinessType', r_new)
            self.assertNotIn('RatingDate', r_new) # Example of a field not kept

        mock_st.success.assert_called_once_with("Processed API response. Identified 2 unique new restaurant records to be added.")

    @patch('data_processing.st')
    def test_empty_master_data_all_api_items_are_new(self, mock_st):
        master_data = []
        # API data can have more fields than ORIGINAL_COLUMNS_TO_KEEP
        api_restaurant = {
            'FHRSID': "1", 'BusinessName': 'Solo Cafe', 'RatingValue': 'Excellent',
            'AddressLine1': 'Addr1', 'PostCode': 'PC', 'LocalAuthorityName': 'LA',
            'NewRatingPending': 'false',
            'ExtraField': 'This will be dropped'
        }
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_restaurant]}}}

        mock_date_str = "mock_date_value"
        with patch('data_processing.datetime') as mock_dt:
            mock_dt.now.return_value.strftime.return_value = mock_date_str
            new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 1)
        r_new = new_restaurants[0]

        self.assertEqual(set(r_new.keys()), set(ORIGINAL_COLUMNS_TO_KEEP))
        self.assertEqual(r_new['FHRSID'], "1")
        self.assertEqual(r_new['BusinessName'], 'Solo Cafe')
        self.assertEqual(r_new['RatingValue'], 'Excellent') # Preserved as per ORIGINAL_COLUMNS_TO_KEEP
        self.assertEqual(r_new['AddressLine1'], 'Addr1')
        self.assertEqual(r_new['PostCode'], 'PC')
        self.assertEqual(r_new['LocalAuthorityName'], 'LA')
        self.assertEqual(r_new['NewRatingPending'], 'false') # Preserved as string
        self.assertEqual(r_new['first_seen'], mock_date_str)
        self.assertEqual(r_new['manual_review'], "not reviewed")
        self.assertIsNone(r_new.get('gemini_insights'))
        # Optional fields from ORIGINAL_COLUMNS_TO_KEEP not provided in API mock
        self.assertIsNone(r_new.get('AddressLine2'))
        self.assertIsNone(r_new.get('AddressLine3'))

        self.assertNotIn('ExtraField', r_new) # Check that extra field is dropped

    @patch('data_processing.st')
    def test_empty_api_data_detail(self, mock_st):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': []}}}
        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        mock_st.info.assert_any_call("API response contained no establishments in 'EstablishmentDetail'.")


    @patch('data_processing.st')
    def test_api_data_establishment_detail_is_none(self, mock_st):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': None}}}
        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        mock_st.warning.assert_called_once_with("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")

    @patch('data_processing.st')
    def test_api_data_missing_establishment_collection(self, mock_st):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {}} # EstablishmentCollection is missing
        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        # This case results in api_establishments = [], which means the code calls:
        # st.info("API response contained no establishments in 'EstablishmentDetail'.")
        # OR if api_establishments is None (which it is not here), it calls st.warning.
        # Since it's an empty list, it should be st.info.
        # If EstablishmentDetail itself was missing, api_establishments would be [].
        # If EstablishmentCollection was missing, api_establishments would be [].
        # If FHRSEstablishment was missing, api_establishments would be [].
        # The logic is:
        # api_establishments = data.get('X', {}).get('Y', {}).get('Z', [])
        # if api_establishments is None: st.warning(...)
        # elif not api_establishments: st.info(...)
        # All these missing key cases lead to api_establishments = [], thus st.info.
        mock_st.info.assert_any_call("API response contained no establishments in 'EstablishmentDetail'.")


    @patch('data_processing.st')
    def test_api_data_missing_fhrestablishment_key(self, mock_st):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {} # FHRSEstablishment key is missing
        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        # Similar to above, this will result in api_establishments = []
        mock_st.info.assert_any_call("API response contained no establishments in 'EstablishmentDetail'.")

    @patch('data_processing.datetime')
    @patch('data_processing.st')
    def test_fhrsid_is_string_after_processing_and_schema_adherence(self, mock_st, mock_datetime):
        """
        Test FHRSID is string after processing, and output adheres to ORIGINAL_COLUMNS_TO_KEEP.
        """
        mock_date_str = "2023-10-27"
        mock_datetime.now.return_value.strftime.return_value = mock_date_str
        master_data = []

        # API data: FHRSID as int/str, BusinessName, some other fields not in ORIGINAL_COLUMNS_TO_KEEP
        api_est_int_fhrsid = {
            'FHRSID': 123, 'BusinessName': 'Testaurant Int',
            'RatingValue': 'Good', 'LocalAuthorityName': 'LA1', 'NewRatingPending': 'false',
            'ExtraInfo': 'will be dropped'
        }
        api_est_str_fhrsid = {
            'FHRSID': "456", 'BusinessName': 'Testaurant Str',
            'AddressLine1': 'Street', 'PostCode': 'PC',
            'RatingValue': 'Bad', 'LocalAuthorityName': 'LA2', 'NewRatingPending': 'TRUE',
            'AnotherExtra': 'also dropped'
        }
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_est_int_fhrsid, api_est_str_fhrsid]}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 2)
        mock_st.success.assert_called_once_with("Processed API response. Identified 2 unique new restaurant records to be added.")

        for r_new in new_restaurants:
            self.assertEqual(set(r_new.keys()), set(ORIGINAL_COLUMNS_TO_KEEP))
            self.assertIsInstance(r_new['FHRSID'], str)
            self.assertEqual(r_new['first_seen'], mock_date_str)
            self.assertEqual(r_new['manual_review'], "not reviewed")
            self.assertIsNone(r_new.get('gemini_insights')) # Default

            # Ensure fields not in ORIGINAL_COLUMNS_TO_KEEP are absent
            self.assertNotIn('ExtraInfo', r_new)
            self.assertNotIn('AnotherExtra', r_new)

            if r_new['BusinessName'] == 'Testaurant Int':
                self.assertEqual(r_new['FHRSID'], "123")
                self.assertEqual(r_new['RatingValue'], 'Good')
                self.assertEqual(r_new['LocalAuthorityName'], 'LA1')
                self.assertEqual(r_new['NewRatingPending'], 'false')
                # Optional fields from ORIGINAL_COLUMNS_TO_KEEP not in API mock for this item
                self.assertIsNone(r_new.get('AddressLine1'))
                self.assertIsNone(r_new.get('AddressLine2'))
                self.assertIsNone(r_new.get('AddressLine3'))
                self.assertIsNone(r_new.get('PostCode'))
            elif r_new['BusinessName'] == 'Testaurant Str':
                self.assertEqual(r_new['FHRSID'], "456")
                self.assertEqual(r_new['RatingValue'], 'Bad')
                self.assertEqual(r_new['LocalAuthorityName'], 'LA2')
                self.assertEqual(r_new['NewRatingPending'], 'TRUE')
                self.assertEqual(r_new['AddressLine1'], 'Street')
                self.assertEqual(r_new['PostCode'], 'PC')
                # Optional fields from ORIGINAL_COLUMNS_TO_KEEP not in API mock for this item
                self.assertIsNone(r_new.get('AddressLine2'))
                self.assertIsNone(r_new.get('AddressLine3'))

    @patch('data_processing.datetime') # Mock datetime for predictable first_seen
    @patch('data_processing.st')      # Mock streamlit
    def test_duplicate_fhrsid_in_api_data_is_added_once(self, mock_st, mock_datetime):
        # Setup mock for datetime.now().strftime()
        mock_datetime_str = "2023-10-28"
        mock_datetime.now.return_value.strftime.return_value = mock_datetime_str

        master_data = [{'FHRSID': "1", 'BusinessName': 'Old Restaurant'}] # One existing unrelated restaurant

        # API data with duplicates and one unique new entry
        # FHRSID "789" is new but appears twice in the API data batch
        # FHRSID "101" is new and appears once
        api_restaurant_duplicate_1 = {
            'FHRSID': "789", 'BusinessName': 'Duplicate Cafe Batch 1',
            'RatingValue': '5', 'AddressLine1': 'Addr D1', 'PostCode': 'PC D1',
            'LocalAuthorityName': 'LA D1', 'NewRatingPending': 'false'
        }
        api_restaurant_duplicate_2 = { # Same FHRSID as above
            'FHRSID': "789", 'BusinessName': 'Duplicate Cafe Batch 2', # Slightly different data for realism
            'RatingValue': '5', 'AddressLine1': 'Addr D2', 'PostCode': 'PC D2',
            'LocalAuthorityName': 'LA D2', 'NewRatingPending': 'false'
        }
        api_restaurant_unique_new = {
            'FHRSID': "101", 'BusinessName': 'Unique New Place',
            'RatingValue': '4', 'AddressLine1': 'Addr U1', 'PostCode': 'PC U1',
            'LocalAuthorityName': 'LA U1', 'NewRatingPending': 'true'
        }

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            api_restaurant_duplicate_1,
            api_restaurant_unique_new,
            api_restaurant_duplicate_2 # Second occurrence of FHRSID "789"
        ]}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 2, "Should identify 2 unique new restaurants.")

        # Extract FHRSIDs from the results for easier checking
        result_fhrsids = {r['FHRSID'] for r in new_restaurants}
        self.assertIn("789", result_fhrsids, "FHRSID 789 should be in the results.")
        self.assertIn("101", result_fhrsids, "FHRSID 101 should be in the results.")

        # Verify that the first occurrence's data for FHRSID "789" was kept
        restaurant_789_data = next((r for r in new_restaurants if r['FHRSID'] == "789"), None)
        self.assertIsNotNone(restaurant_789_data)
        self.assertEqual(restaurant_789_data['BusinessName'], 'Duplicate Cafe Batch 1')
        self.assertEqual(restaurant_789_data['AddressLine1'], 'Addr D1')
        self.assertEqual(restaurant_789_data['first_seen'], mock_datetime_str)
        self.assertEqual(restaurant_789_data['manual_review'], "not reviewed")

        restaurant_101_data = next((r for r in new_restaurants if r['FHRSID'] == "101"), None)
        self.assertIsNotNone(restaurant_101_data)
        self.assertEqual(restaurant_101_data['BusinessName'], 'Unique New Place')
        self.assertEqual(restaurant_101_data['first_seen'], mock_datetime_str)
        self.assertEqual(restaurant_101_data['manual_review'], "not reviewed")

        mock_st.success.assert_called_once_with("Processed API response. Identified 2 unique new restaurant records to be added.")

    @patch('data_processing.datetime')
    @patch('data_processing.st')
    def test_canonical_fhrsid_deduplication_and_non_numeric(self, mock_st, mock_datetime):
        # Setup mock for datetime.now().strftime()
        mock_datetime_str = "2023-11-15"
        mock_datetime.now.return_value.strftime.return_value = mock_datetime_str

        # Define master_data
        master_data = [
            {'FHRSID': 123, 'BusinessName': 'Integer Master', 'AddressLine1': 'Addr Master 1'}, # Will be "123"
            {'FHRSID': "456", 'BusinessName': 'Canonical String Master', 'AddressLine1': 'Addr Master 2'}, # Already "456"
            {'FHRSID': "ABC", 'BusinessName': 'NonNumeric Master', 'AddressLine1': 'Addr Master 3'}, # "ABC", will warn
            {'FHRSID': "M1X", 'BusinessName': 'Malformed Master', 'AddressLine1': 'Addr Master 4'}, # "M1X", will warn
            {'FHRSID': None, 'BusinessName': 'None FHRSID Master', 'AddressLine1': 'Addr Master 5'} # Skipped
        ]

        # Define api_data
        api_establishments = [
            # Duplicates of master_data after canonicalization
            {'FHRSID': "0123", 'BusinessName': 'Integer API Dup', 'AddressLine1': 'Addr API 1'}, # Should become "123"
            {'FHRSID': "456", 'BusinessName': 'Canonical String API Dup', 'AddressLine1': 'Addr API 2'}, # Is "456"
            {'FHRSID': "ABC", 'BusinessName': 'NonNumeric API Dup', 'AddressLine1': 'Addr API 3'}, # Is "ABC", will warn
            {'FHRSID': "M1X", 'BusinessName': 'Malformed API Dup', 'AddressLine1': 'Addr API 4'}, # Is "M1X", will warn
            # New establishments
            {'FHRSID': "0789", 'BusinessName': 'New Numeric Normalized', 'AddressLine1': 'Addr API 5'}, # Becomes "789"
            {'FHRSID': "DEF", 'BusinessName': 'New NonNumeric', 'AddressLine1': 'Addr API 6'}, # Is "DEF", will warn
            {'FHRSID': "A2Y", 'BusinessName': 'New Malformed API', 'AddressLine1': 'Addr API 7'}, # Is "A2Y", will warn
            # Skipped API entry
            {'FHRSID': None, 'BusinessName': 'None FHRSID API', 'AddressLine1': 'Addr API 8'}
        ]
        # Add other required fields from ORIGINAL_COLUMNS_TO_KEEP to all api_establishments for simplicity
        for est_api in api_establishments:
            if est_api['FHRSID'] is not None: # Only add to valid entries for processing
                est_api.update({key: None for key in ORIGINAL_COLUMNS_TO_KEEP if key not in est_api})

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 3, "Should identify 3 new unique restaurants.")

        added_fhrsids = sorted([r['FHRSID'] for r in new_restaurants])
        expected_fhrsids = sorted(["789", "DEF", "A2Y"])
        self.assertEqual(added_fhrsids, expected_fhrsids, "FHRSIDs of new restaurants should be the canonical forms.")

        # Check data for one of the new restaurants to ensure fields are set
        new_def = next(r for r in new_restaurants if r['FHRSID'] == "DEF")
        self.assertEqual(new_def['BusinessName'], 'New NonNumeric')
        self.assertEqual(new_def['first_seen'], mock_datetime_str)
        self.assertEqual(new_def['manual_review'], "not reviewed")

        # Verify st.warning calls
        # Expected warnings:
        # 1. Master: "ABC" (int("ABC") fails)
        # 2. Master: "M1X" (int("M1X") fails)
        # 3. API: "ABC" (int("ABC") fails during its canonicalization)
        # 4. API: "M1X" (int("M1X") fails during its canonicalization)
        # 5. API: "DEF" (int("DEF") fails during its canonicalization)
        # 6. API: "A2Y" (int("A2Y") fails during its canonicalization)

        # Note: The FHRSID "0123" from API becomes "123" without warning.
        # FHRSID "456" from API is already canonical and numeric, no warning.

        expected_warning_calls = [
            unittest.mock.call("FHRSID 'ABC' from master_data could not be converted to int. Using original string value for comparison."),
            unittest.mock.call("FHRSID 'M1X' from master_data could not be converted to int. Using original string value for comparison."),
            unittest.mock.call("FHRSID 'ABC' from API data could not be converted to int. Using original string value."),
            unittest.mock.call("FHRSID 'M1X' from API data could not be converted to int. Using original string value."),
            unittest.mock.call("FHRSID 'DEF' from API data could not be converted to int. Using original string value."),
            unittest.mock.call("FHRSID 'A2Y' from API data could not be converted to int. Using original string value.")
        ]

        # Check if all expected calls are present, regardless of order for warnings from the same source (master/api)
        # However, the order of master data warnings should precede api data warnings.
        # And within API data, the order should be as per api_establishments list.
        # So, a direct comparison of call_args_list is better.
        mock_st.warning.assert_has_calls(expected_warning_calls, any_order=False)
        self.assertEqual(mock_st.warning.call_count, 6, "Expected 6 warning calls.")

        # Also check success message for the correct count
        mock_st.success.assert_called_once_with("Processed API response. Identified 3 unique new restaurant records to be added.")

    @patch('data_processing.datetime')
    @patch('data_processing.st')
    def test_deduplication_with_corrected_fhrsid_key(self, mock_st, mock_datetime):
        mock_date_str = "2024-01-01"
        mock_datetime.now.return_value.strftime.return_value = mock_date_str

        # Master data uses lowercase 'fhrsid'
        master_data = [
            {'fhrsid': "123", 'BusinessName': 'BQ Cafe Old'}, # Numeric FHRSID
            {'fhrsid': "ABC", 'BusinessName': 'BQ NonNumeric Old'} # Non-numeric FHRSID
            # Other ORIGINAL_COLUMNS_TO_KEEP fields are not strictly necessary for master_data
            # in the context of FHRSID matching, as the function only uses 'fhrsid'.
        ]

        # API data uses 'FHRSID' and includes all ORIGINAL_COLUMNS_TO_KEEP for new items
        api_establishments = [
            # Matches master_data 'fhrsid': "123"
            {'FHRSID': "123", 'BusinessName': 'API Cafe Update', 'RatingValue': '3', 'NewRatingPending': 'false',
             'AddressLine1': 'Addr1', 'AddressLine2': None, 'AddressLine3': None, 'PostCode': 'PC1',
             'LocalAuthorityName': 'LA1', 'gemini_insights': None},
            # New numeric FHRSID
            {'FHRSID': "789", 'BusinessName': 'API Cafe New Numeric', 'RatingValue': '5', 'NewRatingPending': 'false',
             'AddressLine1': 'Addr2', 'AddressLine2': 'Suite B', 'AddressLine3': None, 'PostCode': 'PC2',
             'LocalAuthorityName': 'LA2', 'gemini_insights': 'Good place'},
            # Matches master_data 'fhrsid': "ABC"
            {'FHRSID': "ABC", 'BusinessName': 'API NonNumeric Update', 'RatingValue': '2', 'NewRatingPending': 'true',
             'AddressLine1': 'Addr3', 'AddressLine2': None, 'AddressLine3': 'Old Town', 'PostCode': 'PC3',
             'LocalAuthorityName': 'LA3', 'gemini_insights': None},
            # New non-numeric FHRSID
            {'FHRSID': "XYZ", 'BusinessName': 'API Cafe New NonNumeric', 'RatingValue': '1', 'NewRatingPending': 'true',
             'AddressLine1': 'Addr4', 'AddressLine2': None, 'AddressLine3': None, 'PostCode': 'PC4',
             'LocalAuthorityName': 'LA4', 'gemini_insights': None}
        ]
        # Ensure all ORIGINAL_COLUMNS_TO_KEEP are present if not already defined for API items
        for est_api in api_establishments:
            for key in ORIGINAL_COLUMNS_TO_KEEP:
                if key not in est_api: # Set missing keys to None for realistic processing by the function
                    est_api[key] = None
            # Ensure FHRSID is present as it's the primary key for matching
            if 'FHRSID' not in est_api: # Should not happen with above data, but good check
                est_api['FHRSID'] = None


        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 2, "Should identify 2 new unique restaurants.")

        added_fhrsids = sorted([r['FHRSID'] for r in new_restaurants])
        expected_fhrsids = sorted(["789", "XYZ"])
        self.assertEqual(added_fhrsids, expected_fhrsids)

        expected_new_numeric = {col: None for col in ORIGINAL_COLUMNS_TO_KEEP}
        expected_new_numeric.update({
            'FHRSID': "789", 'BusinessName': 'API Cafe New Numeric', 'RatingValue': '5',
            'NewRatingPending': 'false', 'first_seen': mock_date_str, 'manual_review': "not reviewed",
            'AddressLine1': 'Addr2', 'AddressLine2': 'Suite B', 'PostCode': 'PC2',
            'LocalAuthorityName': 'LA2', 'gemini_insights': 'Good place'
        })

        expected_new_non_numeric = {col: None for col in ORIGINAL_COLUMNS_TO_KEEP}
        expected_new_non_numeric.update({
            'FHRSID': "XYZ", 'BusinessName': 'API Cafe New NonNumeric', 'RatingValue': '1',
            'NewRatingPending': 'true', 'first_seen': mock_date_str, 'manual_review': "not reviewed",
            'AddressLine1': 'Addr4', 'PostCode': 'PC4', 'LocalAuthorityName': 'LA4'
        })

        # Check details of the new restaurants
        for r_new in new_restaurants:
            if r_new['FHRSID'] == "789":
                self.assertEqual(r_new, expected_new_numeric)
            elif r_new['FHRSID'] == "XYZ":
                self.assertEqual(r_new, expected_new_non_numeric)

        # Assert st.warning calls
        # 1. Master: "ABC" (int(est['fhrsid']) fails)
        # 2. API: "ABC" (int(original_api_fhrsid) fails for its canonicalization)
        # 3. API: "XYZ" (int(original_api_fhrsid) fails for its canonicalization)
        expected_warning_calls = [
            unittest.mock.call("FHRSID 'ABC' from master_data could not be converted to int. Using original string value for comparison."),
            unittest.mock.call("FHRSID 'ABC' from API data could not be converted to int. Using original string value."),
            unittest.mock.call("FHRSID 'XYZ' from API data could not be converted to int. Using original string value.")
        ]
        # Use assert_has_calls which allows for other calls in between, or check call_count and specific calls
        mock_st.warning.assert_has_calls(expected_warning_calls, any_order=False) # Order should be preserved here
        self.assertEqual(mock_st.warning.call_count, 3, "Expected 3 warning calls for non-numeric FHRSIDs.")

        mock_st.success.assert_called_once_with("Processed API response. Identified 2 unique new restaurant records to be added.")


if __name__ == '__main__':
    unittest.main()

# --- Tests for load_data_from_csv ---
from data_processing import load_data_from_csv # Already imported at top but good for section visibility
import io # For io.StringIO

class TestLoadDataFromCsv(unittest.TestCase):
    @patch('data_processing.st.error') # Mock st.error from data_processing module
    def test_successful_load(self, mock_st_error):
        csv_content = '"fhrsid","colA"\n"1","abc"\n"2","def"'
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['fhrsid', 'colA'])
        pd.testing.assert_series_equal(df['fhrsid'], pd.Series(["1", "2"], name='fhrsid', dtype=str))
        mock_st_error.assert_not_called()

    @patch('data_processing.st.error')
    def test_missing_fhrsid_column(self, mock_st_error):
        csv_content = '"colX","colA"\n"1","abc"'
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNone(df)
        mock_st_error.assert_called_once_with("The required 'fhrsid' column is missing in the uploaded CSV file.")

    @patch('data_processing.st.error')
    def test_empty_csv_file_content(self, mock_st_error):
        # This simulates a file that was uploaded but its content is empty, leading to EmptyDataError
        csv_content = ""
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNone(df)
        mock_st_error.assert_called_once_with("The uploaded CSV file is empty or contains no data.")

    @patch('data_processing.st.error')
    def test_empty_csv_file_just_headers(self, mock_st_error):
        # CSV with only headers, no data rows
        csv_content = '"fhrsid","colA"'
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNotNone(df)
        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), ['fhrsid', 'colA']) # Columns should still be there
        # The function `load_data_from_csv` itself has a check for `df.empty` after read_csv
        # and returns this empty DataFrame. It doesn't call st.error in this specific case.
        # However, the initial implementation in the prompt for load_data_from_csv was:
        # "if df.empty: st.error("The uploaded CSV file is empty."); return None"
        # The implemented code for load_data_from_csv in data_processing.py is:
        # "if df.empty: st.error("The uploaded CSV file is empty."); return None"
        # This means this test case should expect st.error and None.
        # Self-correction based on actual implementation of load_data_from_csv:
        # It should return None and call st.error.
        # Let's re-verify the implementation in `data_processing.py` for `load_data_from_csv`:
        # try:
        #   df = pd.read_csv(uploaded_file)
        #   if df.empty:  <-- This is after successful read_csv
        #     st.error("The uploaded CSV file is empty.")
        #     return None
        # This test should assert st.error was called and df is None.
        self.assertIsNone(df)
        mock_st_error.assert_called_once_with("The uploaded CSV file is empty.")


    @patch('data_processing.st.error')
    def test_case_insensitive_fhrsid(self, mock_st_error):
        csv_content = '"FHRSID","colA"\n"1","abc"'
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNotNone(df)
        self.assertIn('fhrsid', df.columns) # Should be renamed to lowercase 'fhrsid'
        self.assertTrue(pd.api.types.is_string_dtype(df['fhrsid']))
        self.assertEqual(df['fhrsid'].iloc[0], "1")
        mock_st_error.assert_not_called()

    @patch('data_processing.st.error')
    def test_parser_error_malformed_csv(self, mock_st_error):
        # Malformed CSV (e.g., inconsistent number of columns per row after header)
        csv_content = '"fhrsid","colA"\n"1"' # Second row has only one value
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNone(df)
        mock_st_error.assert_called_once_with("Error parsing the CSV file. Please ensure it's a valid CSV format.")

    @patch('data_processing.st.error')
    def test_fhrsid_column_present_but_empty_values(self, mock_st_error):
        csv_content = '"fhrsid","colA"\n"","abc"\n"","def"' # fhrsid values are empty strings
        simulated_file = io.StringIO(csv_content)

        df = load_data_from_csv(simulated_file)

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        # load_data_from_csv converts fhrsid to string. Empty strings are valid strings.
        pd.testing.assert_series_equal(df['fhrsid'], pd.Series(["", ""], name='fhrsid', dtype=str))
        mock_st_error.assert_not_called()

    @patch('data_processing.st.error')
    def test_generic_exception_during_read(self, mock_st_error):
        simulated_file = MagicMock()
        simulated_file.read.side_effect = Exception("Unexpected read error")

        # We need to ensure pd.read_csv gets this mock.
        # This is tricky because pd.read_csv takes the file object directly.
        # We'll patch pd.read_csv itself for this one case.
        with patch('data_processing.pd.read_csv', side_effect=Exception("Simulated pandas error")):
            df = load_data_from_csv(simulated_file)

        self.assertIsNone(df)
        mock_st_error.assert_called_once_with("An unexpected error occurred while reading the CSV file: Simulated pandas error")
