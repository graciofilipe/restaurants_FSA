import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

# Import the module to be tested (will be created)
try:
    from scripts.update_search_config import add_search_config
except ImportError:
    pass

class TestUpdateSearchConfig(unittest.TestCase):
    
    @patch('google.cloud.bigquery.Client')
    def test_add_search_config_success(self, mock_client_cls):
        # Setup mock
        mock_client = mock_client_cls.return_value
        mock_table = MagicMock()
        mock_client.get_table.return_value = mock_table
        mock_client.insert_rows_from_dataframe.return_value = [] # Success returns empty list of errors

        # Test data
        coords = [(0.1, 51.5)] # Lon, Lat
        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "config_table"
        
        # Execute
        result = add_search_config(project_id, dataset_id, table_id, coords)
        
        # Verify
        self.assertTrue(result)
        mock_client.insert_rows_from_dataframe.assert_called_once()
        
        # Verify the dataframe passed to insert
        call_args = mock_client.insert_rows_from_dataframe.call_args
        df_arg = call_args[0][1]
        self.assertIsInstance(df_arg, pd.DataFrame)
        self.assertEqual(len(df_arg), 1)
        self.assertEqual(df_arg.iloc[0]['longitude'], 0.1)
        self.assertEqual(df_arg.iloc[0]['latitude'], 51.5)

    @patch('google.cloud.bigquery.Client')
    def test_add_search_config_failure(self, mock_client_cls):
        # Setup mock failure
        mock_client = mock_client_cls.return_value
        mock_client.insert_rows_from_dataframe.return_value = [{"index": 0, "errors": ["some error"]}]

        # Test data
        coords = [(0.1, 51.5)]
        
        # Execute
        result = add_search_config("p", "d", "t", coords)
        
        # Verify
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
