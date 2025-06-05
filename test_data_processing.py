import unittest
from unittest.mock import MagicMock

# Assuming data_processing.py is in the same directory or accessible in PYTHONPATH
from data_processing import load_master_data, process_and_update_master_data

class TestLoadMasterData(unittest.TestCase):

    def test_manual_review_added_if_missing_or_none(self):
        """
        Tests that 'manual_review' is added if missing or None.
        """
        mock_data = [
            {"id": 1, "name": "Restaurant A"}, # Missing manual_review
            {"id": 2, "name": "Restaurant B", "manual_review": None}, # manual_review is None
            {"id": 3, "name": "Restaurant C", "manual_review": "already reviewed"},
            {"id": 4, "name": "Restaurant D", "manual_review": False}
        ]
        expected_data = [
            {"id": 1, "name": "Restaurant A", "manual_review": "not reviewed"},
            {"id": 2, "name": "Restaurant B", "manual_review": "not reviewed"},
            {"id": 3, "name": "Restaurant C", "manual_review": "already reviewed"},
            {"id": 4, "name": "Restaurant D", "manual_review": False}
        ]

        mock_load_json_func = MagicMock(return_value=mock_data)

        # Mock st.info, st.warning, st.success used within load_master_data
        with unittest.mock.patch('data_processing.st') as mock_st:
            result = load_master_data("dummy_uri", mock_load_json_func)

        self.assertEqual(result, expected_data)
        mock_load_json_func.assert_called_once_with("dummy_uri")

    def test_manual_review_preserved_if_exists(self):
        """
        Tests that existing 'manual_review' values (other than None) are preserved.
        """
        mock_data = [
            {"id": 1, "name": "Restaurant A", "manual_review": "human verified"},
            {"id": 2, "name": "Restaurant B", "manual_review": ""}, # Empty string is a value
        ]
        expected_data = [
            {"id": 1, "name": "Restaurant A", "manual_review": "human verified"},
            {"id": 2, "name": "Restaurant B", "manual_review": ""},
        ]
        mock_load_json_func = MagicMock(return_value=mock_data)

        with unittest.mock.patch('data_processing.st') as mock_st:
            result = load_master_data("dummy_uri", mock_load_json_func)

        self.assertEqual(result, expected_data)

    def test_empty_data_handling(self):
        """
        Tests handling of empty input data.
        """
        mock_load_json_func = MagicMock(return_value=[])
        with unittest.mock.patch('data_processing.st') as mock_st:
            result = load_master_data("dummy_uri", mock_load_json_func)
        self.assertEqual(result, [])

    def test_non_list_data_handling(self):
        """
        Tests handling of data that is not a list (should return empty list).
        """
        mock_load_json_func = MagicMock(return_value={"key": "value"}) # Not a list
        with unittest.mock.patch('data_processing.st') as mock_st:
            result = load_master_data("dummy_uri", mock_load_json_func)
        self.assertEqual(result, [])
        mock_st.warning.assert_called()

class TestProcessAndUpdateMasterData(unittest.TestCase):

    def test_new_establishments_get_manual_review(self):
        """
        Tests that new establishments added from API data get 'manual_review': 'not reviewed'.
        """
        master_data = [
            {"FHRSID": "100", "name": "Existing Place", "manual_review": "previously reviewed"}
        ]
        api_establishments_details = [
            {"FHRSID": "101", "name": "New Place A"}, # New, no manual_review
            {"FHRSID": "102", "name": "New Place B", "manual_review": "some initial value"} # New, but will be overwritten
        ]
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments_details}}}

        # Mock st.success used within process_and_update_master_data
        with unittest.mock.patch('data_processing.st') as mock_st:
            updated_master, new_additions = process_and_update_master_data(list(master_data), api_data)

        self.assertEqual(new_additions, 2)
        self.assertEqual(len(updated_master), 3)

        found_new_a = False
        found_new_b = False
        for item in updated_master:
            if item["FHRSID"] == "101":
                found_new_a = True
                self.assertEqual(item["manual_review"], "not reviewed")
                self.assertTrue("first_seen" in item)
            elif item["FHRSID"] == "102":
                found_new_b = True
                self.assertEqual(item["manual_review"], "not reviewed") # Even if it had a value, it's a new record
                self.assertTrue("first_seen" in item)
            elif item["FHRSID"] == "100":
                self.assertEqual(item["manual_review"], "previously reviewed") # Existing preserved

        self.assertTrue(found_new_a)
        self.assertTrue(found_new_b)

    def test_existing_establishments_manual_review_preserved(self):
        """
        Tests that 'manual_review' of existing establishments is preserved.
        """
        master_data = [
            {"FHRSID": "100", "name": "Old Place", "manual_review": "verified", "first_seen": "2023-01-01"}
        ]
        # API data contains the same establishment, but process_and_update should not re-add or modify it
        # if the FHRSID matches an existing one.
        api_establishments_details = [
            {"FHRSID": "100", "name": "Old Place Updated Name?", "manual_review": "changed in api"}
        ]
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments_details}}}

        with unittest.mock.patch('data_processing.st') as mock_st:
            updated_master, new_additions = process_and_update_master_data(list(master_data), api_data)

        self.assertEqual(new_additions, 0) # No new record should be added
        self.assertEqual(len(updated_master), 1)
        self.assertEqual(updated_master[0]["FHRSID"], "100")
        self.assertEqual(updated_master[0]["name"], "Old Place") # Name should not be updated by this function
        self.assertEqual(updated_master[0]["manual_review"], "verified") # manual_review preserved

    def test_no_new_establishments(self):
        """
        Tests behavior when API data has no new establishments.
        """
        master_data = [
            {"FHRSID": "100", "name": "Existing Place", "manual_review": "reviewed"}
        ]
        api_establishments_details = [
            {"FHRSID": "100", "name": "Existing Place From API"} # Same FHRSID
        ]
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments_details}}}

        with unittest.mock.patch('data_processing.st') as mock_st:
            updated_master, new_additions = process_and_update_master_data(list(master_data), api_data)

        self.assertEqual(new_additions, 0)
        self.assertEqual(len(updated_master), 1)
        self.assertEqual(updated_master[0]["manual_review"], "reviewed")

    def test_empty_api_data(self):
        """
        Tests behavior with empty 'EstablishmentDetail' in API data.
        """
        master_data = [{"FHRSID": "100", "name": "Place", "manual_review": "ok"}]
        api_data_empty = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': []}}}

        with unittest.mock.patch('data_processing.st') as mock_st:
            updated_master, new_additions = process_and_update_master_data(list(master_data), api_data_empty)

        self.assertEqual(new_additions, 0)
        self.assertEqual(len(updated_master), 1)
        self.assertEqual(updated_master, master_data) # Should be unchanged
        mock_st.info.assert_called() # Or warning, depending on implementation details for empty lists

    def test_api_data_missing_keys(self):
        """
        Tests robustness if API data structure is incomplete.
        """
        master_data = [{"FHRSID": "100", "name": "Place", "manual_review": "ok"}]
        api_data_malformed1 = {} # Missing FHRSEstablishment
        api_data_malformed2 = {'FHRSEstablishment': {}} # Missing EstablishmentCollection
        api_data_malformed3 = {'FHRSEstablishment': {'EstablishmentCollection': {}}} # Missing EstablishmentDetail

        with unittest.mock.patch('data_processing.st') as mock_st:
            updated_master1, new_additions1 = process_and_update_master_data(list(master_data), api_data_malformed1)
            self.assertEqual(new_additions1, 0)
            self.assertEqual(updated_master1, master_data)
            # Check for st.warning or st.info indicating no data was processed
            # Example: mock_st.warning.assert_any_call("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")

            updated_master2, new_additions2 = process_and_update_master_data(list(master_data), api_data_malformed2)
            self.assertEqual(new_additions2, 0)
            self.assertEqual(updated_master2, master_data)

            updated_master3, new_additions3 = process_and_update_master_data(list(master_data), api_data_malformed3)
            self.assertEqual(new_additions3, 0)
            self.assertEqual(updated_master3, master_data)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
