import unittest
from unittest.mock import MagicMock, patch
import sys

# 1. Mock streamlit
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st
sys.modules['streamlit.components.v1'] = MagicMock()

# 2. Mock google libraries (just enough for st_app.py imports)
mock_google = MagicMock()
sys.modules['google'] = mock_google
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.bigquery'] = MagicMock()

# 3. Mock app.services.bq_utils to prevent it from importing pandas_gbq etc.
mock_bq_utils = MagicMock()
sys.modules['app.services.bq_utils'] = mock_bq_utils

# 4. Mock app.services.api_client
sys.modules['app.services.api_client'] = MagicMock()

# Now import the module under test
from app.ui.st_app import handle_fetch_data_action

class TestStAppLogic(unittest.TestCase):
    @patch('app.ui.st_app.parse_coordinates')
    @patch('app.ui.st_app.run_data_synchronization')
    @patch('app.ui.st_app.parse_bq_path')
    def test_handle_fetch_data_action_uses_core_functions(self, mock_parse_bq, mock_sync, mock_parse_coords):
        # Setup
        mock_parse_coords.return_value = ([(1.0, 2.0)], []) # Valid coords, no errors
        mock_parse_bq.return_value = ("p", "d", "t")
        
        # mock_sync returns (master_data, new_restaurants, summary_msg)
        mock_sync.return_value = ([], [], "No new data")

        # Execute
        handle_fetch_data_action("1.0, 2.0", 10, "p.d.t")

        # Verify
        mock_parse_coords.assert_called_once_with("1.0, 2.0")
        mock_parse_bq.assert_called_once_with("p.d.t")
        mock_sync.assert_called_once_with([(1.0, 2.0)], 10, "p", "d", "t")

if __name__ == '__main__':
    unittest.main()