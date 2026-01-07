
import unittest
from unittest.mock import MagicMock, patch
from app.cron import fetch_weekly

class TestFetchWeekly(unittest.TestCase):

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_main_loop_iteration(self, mock_run_sync, mock_get_config):
        # Mock 2 config rows
        mock_get_config.return_value = [
            {'latitude': 51.5, 'longitude': -0.1, 'max_results': 100},
            {'latitude': 52.5, 'longitude': -0.2, 'max_results': 100}
        ]
        
        fetch_weekly.main()
        
        # Verify run_sync_for_config called twice
        self.assertEqual(mock_run_sync.call_count, 2)

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_main_loop_continues_on_error(self, mock_run_sync, mock_get_config):
        # Mock 2 config rows
        mock_get_config.return_value = [
            {'latitude': 51.5, 'longitude': -0.1},
            {'latitude': 52.5, 'longitude': -0.2}
        ]
        
        # First call raises exception, second succeeds
        mock_run_sync.side_effect = [Exception("Test Error"), None]
        
        # Should not raise exception
        fetch_weekly.main()
        
        # Verify run_sync_for_config called twice
        self.assertEqual(mock_run_sync.call_count, 2)

    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_master_data')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_run_sync_for_config_new_schema(self, mock_append, mock_process, mock_load, mock_fetch):
        # Config with new schema
        config = {
            'latitude': 51.5074, 
            'longitude': -0.1278, 
            'max_results': 5000, 
            'target_bq_table': 'p.d.t',
            'radius': 5
        }
        
        # Mock returns
        mock_fetch.return_value = [] # Return empty list to stop early or simple list
        mock_load.return_value = []
        mock_process.return_value = ([], "Summary")
        
        fetch_weekly.run_sync_for_config(config)
        
        # Verify fetch_data_for_all_coordinates called with correct tuple
        # Expected: [(lon, lat)]
        mock_fetch.assert_called_with([(-0.1278, 51.5074)], 5000)

if __name__ == '__main__':
    unittest.main()
