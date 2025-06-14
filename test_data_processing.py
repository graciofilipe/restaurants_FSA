import unittest # Changed from pytest to unittest for consistency with TestAppendToBigQuery
from unittest.mock import MagicMock, patch
from data_processing import load_master_data, process_and_update_master_data, load_json_from_local_file_path # Added load_json_from_local_file_path
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
        mock_st.info.assert_called_once_with("Processed API response. No new restaurant records identified.")

    @patch('data_processing.datetime') # Mock datetime for predictable first_seen
    @patch('data_processing.st') # Mock streamlit
    def test_add_new_restaurants_and_fields_initialization(self, mock_st, mock_datetime):
        # Setup mock for datetime.now().strftime()
        mock_datetime.now.return_value.strftime.return_value = "2023-10-26"

        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        # Define API data with one existing and two new restaurants
        api_restaurant_1_existing = {'FHRSID': "1", 'name': 'A_updated'} # FHRSID is string
        api_restaurant_2_new = {'FHRSID': "2", 'name': 'B', 'RatingValue': '5'} # FHRSID is string
        api_restaurant_3_new = {'FHRSID': "3", 'name': 'C'} # FHRSID is string

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            api_restaurant_1_existing,
            api_restaurant_2_new,
            api_restaurant_3_new
        ]}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 2)

        # Check if the new restaurants are the correct ones (order might not be guaranteed)
        fhrsid_in_new = [r['FHRSID'] for r in new_restaurants]
        self.assertIn("2", fhrsid_in_new) # Expect string FHRSID
        self.assertIn("3", fhrsid_in_new) # Expect string FHRSID

        # Check properties of the new restaurants
        for r_new in new_restaurants:
            self.assertEqual(r_new['first_seen'], "2023-10-26")
            self.assertEqual(r_new['manual_review'], "not reviewed")
            if r_new['FHRSID'] == "2": # api_restaurant_2_new, compare with string
                self.assertEqual(r_new['name'], 'B')
                self.assertEqual(r_new['RatingValue'], '5') # Ensure other fields preserved
            elif r_new['FHRSID'] == "3": # api_restaurant_3_new, compare with string
                 self.assertEqual(r_new['name'], 'C')

        mock_st.success.assert_called_once_with("Processed API response. Identified 2 new restaurant records to be added.")

    @patch('data_processing.st')
    def test_empty_master_data_all_api_items_are_new(self, mock_st):
        master_data = []
        api_restaurant = {'FHRSID': "1", 'name': 'A'} # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_restaurant]}}}

        with patch('data_processing.datetime') as mock_dt: # Mock datetime for first_seen
            mock_dt.now.return_value.strftime.return_value = "mock_date"
            new_restaurants = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 1)
        self.assertEqual(new_restaurants[0]['FHRSID'], "1") # Expect string FHRSID
        self.assertEqual(new_restaurants[0]['first_seen'], "mock_date")
        self.assertEqual(new_restaurants[0]['manual_review'], "not reviewed")

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
    def test_fhrsid_is_string_after_processing(self, mock_st, mock_datetime): # Renamed test
        """
        Test that FHRSID is a string after processing, regardless of original type from API.
        """
        # Setup mock for datetime.now().strftime()
        mock_datetime.now.return_value.strftime.return_value = "2023-10-27"

        master_data = [] # No existing master data

        # API data with FHRSID as integer and string
        api_establishment_int_fhrsid = {'FHRSID': 123, 'name': 'Testaurant'}
        api_establishment_str_fhrsid = {'FHRSID': "456", 'name': 'Another Testaurant'}

        api_data = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [api_establishment_int_fhrsid, api_establishment_str_fhrsid]
                }
            }
        }

        new_restaurants = process_and_update_master_data(master_data, api_data)

        # Assertions
        self.assertEqual(len(new_restaurants), 2, "Should find two new restaurants")
        mock_st.success.assert_called_once_with("Processed API response. Identified 2 new restaurant records to be added.")

        # Check first restaurant (originally int FHRSID from API)
        added_restaurant_1 = next(r for r in new_restaurants if r['name'] == 'Testaurant')
        self.assertIsInstance(added_restaurant_1['FHRSID'], str, "FHRSID should be a string")
        self.assertEqual(added_restaurant_1['FHRSID'], "123", "FHRSID should be the string '123'")
        self.assertEqual(added_restaurant_1['name'], 'Testaurant')
        self.assertEqual(added_restaurant_1['first_seen'], "2023-10-27")
        self.assertEqual(added_restaurant_1['manual_review'], "not reviewed")

        # Check second restaurant (originally string FHRSID from API)
        added_restaurant_2 = next(r for r in new_restaurants if r['name'] == 'Another Testaurant')
        self.assertIsInstance(added_restaurant_2['FHRSID'], str, "FHRSID should be a string")
        self.assertEqual(added_restaurant_2['FHRSID'], "456", "FHRSID should be the string '456'")
        self.assertEqual(added_restaurant_2['name'], 'Another Testaurant')
        self.assertEqual(added_restaurant_2['first_seen'], "2023-10-27")
        self.assertEqual(added_restaurant_2['manual_review'], "not reviewed")


if __name__ == '__main__':
    unittest.main()
