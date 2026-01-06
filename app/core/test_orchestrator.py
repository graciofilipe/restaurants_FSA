import unittest
from unittest.mock import MagicMock, patch
from typing import List, Tuple, Dict, Any

# We will implement these in app/core/data_processing.py
from app.core.data_processing import run_data_synchronization, parse_bq_path

class TestOrchestrator(unittest.TestCase):
    
    def test_parse_bq_path_valid(self):
        project, dataset, table = parse_bq_path("my-project.my_dataset.my_table")
        self.assertEqual(project, "my-project")
        self.assertEqual(dataset, "my_dataset")
        self.assertEqual(table, "my_table")

    def test_parse_bq_path_invalid_format(self):
        with self.assertRaises(ValueError):
            parse_bq_path("invalid_path")
        with self.assertRaises(ValueError):
            parse_bq_path("project.dataset") # missing table

    @patch('app.core.data_processing.fetch_data_for_all_coordinates')
    @patch('app.core.data_processing.load_master_data')
    @patch('app.core.data_processing.process_and_update_master_data')
    def test_run_data_synchronization_success(self, mock_process, mock_load, mock_fetch):
        # Setup mocks
        valid_coords = [(1.0, 1.0)]
        max_results = 10
        project_id = "p"
        dataset_id = "d"
        table_id = "t"

        mock_fetch.return_value = [{'FHRSID': '1', 'name': 'New Place'}]
        mock_load.return_value = [{'FHRSID': '2', 'name': 'Old Place'}]
        mock_process.return_value = ([{'FHRSID': '1', 'name': 'New Place'}], "Found 1 new")

        # Execute
        master_data, new_restaurants, summary_msg = run_data_synchronization(
            valid_coords, max_results, project_id, dataset_id, table_id
        )

        # Assertions
        mock_fetch.assert_called_once_with(valid_coords, max_results)
        
        # Verify construction of combined_api_data passed to process
        expected_api_data = {
            'FHRSEstablishment': {
                'EstablishmentCollection': {
                    'EstablishmentDetail': [{'FHRSID': '1', 'name': 'New Place'}]
                }
            }
        }
        
        mock_load.assert_called_once() # Args are verified in load_master_data tests, but we can check if needed
        mock_process.assert_called_once_with(mock_load.return_value, expected_api_data)
        
        self.assertEqual(master_data, mock_load.return_value)
        self.assertEqual(new_restaurants, mock_process.return_value[0])
        self.assertEqual(summary_msg, "Found 1 new")

    @patch('app.core.data_processing.fetch_data_for_all_coordinates')
    @patch('app.core.data_processing.load_master_data')
    def test_run_data_synchronization_load_master_fails(self, mock_load, mock_fetch):
        # Setup failure
        mock_load.side_effect = Exception("BQ Error")
        
        with self.assertRaises(Exception):
            run_data_synchronization([], 10, "p", "d", "t")

if __name__ == '__main__':
    unittest.main()
