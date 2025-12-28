import unittest
from unittest.mock import MagicMock, patch
from data_processing import process_and_update_master_data
from bq_utils import ORIGINAL_COLUMNS_TO_KEEP

class TestRemoveMapsUrl(unittest.TestCase):
    @patch('data_processing.datetime')
    @patch('data_processing.st')
    def test_maps_link_removed(self, mock_st, mock_datetime):
        """
        Test that 'Maps Link' is NOT added to processed restaurants.
        This test is expected to FAIL before the implementation change.
        """
        mock_date_str = "2023-10-27"
        mock_datetime.now.return_value.strftime.return_value = mock_date_str
        master_data = []

        api_est = {
            'FHRSID': 123, 'BusinessName': 'Testaurant',
            'AddressLine1': '123 Main St', 'PostCode': 'A1 1AA',
            'RatingValue': '5', 'LocalAuthorityName': 'LA', 'NewRatingPending': 'false'
        }
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_est]}}}

        new_restaurants = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 1)
        r_new = new_restaurants[0]
        
        # This assertion is expected to fail initially
        self.assertNotIn('Maps Link', r_new, "'Maps Link' should not be present in the processed data")

if __name__ == '__main__':
    unittest.main()
