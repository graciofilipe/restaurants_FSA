
import unittest
from unittest.mock import MagicMock, patch
from google.cloud import bigquery
import scripts.setup_config_table as setup_script

class TestSetupConfigData(unittest.TestCase):

    @patch('scripts.setup_config_table.bigquery.Client')
    def test_populate_with_multiple_coordinates(self, mock_client_cls):
        # Setup mock
        mock_client = mock_client_cls.return_value
        
        # Mock data to be inserted
        test_coordinates = [
            {"lat": 51.5074, "lon": -0.1278}, # London
            {"lat": 53.4808, "lon": -2.2426}, # Manchester
        ]
        
        # Override the hardcoded coordinates in the script for testing if possible
        # Or better, refactor the script to accept an argument.
        # For this test, assuming we refactor `create_config_table` to accept data.
        
        setup_script.create_config_table(initial_coordinates=test_coordinates)
        
        # Verify insert_rows_json was called
        mock_client.insert_rows_json.assert_called()
        
        # Check the data passed to insert_rows_json
        call_args = mock_client.insert_rows_json.call_args
        table_ref = call_args[0][0]
        rows_inserted = call_args[0][1]
        
        self.assertEqual(len(rows_inserted), 2)
        self.assertEqual(rows_inserted[0]['latitude'], 51.5074)
        self.assertEqual(rows_inserted[0]['longitude'], -0.1278)
        self.assertEqual(rows_inserted[0]['max_results'], 5000) # Check default
        self.assertEqual(rows_inserted[1]['latitude'], 53.4808)
        
if __name__ == '__main__':
    unittest.main()
