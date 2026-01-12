import unittest
from unittest.mock import MagicMock, patch
from app.ui.st_app import main_ui

class SessionState(dict):
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

class TestStAppRefresh(unittest.TestCase):
    def setUp(self):
        self.session_state = SessionState()

    @patch('app.ui.st_app.st')
    @patch('app.ui.st_app.get_distinct_local_authorities')
    @patch('app.ui.st_app.parse_bq_path')
    def test_refresh_authorities_button(self, mock_parse_bq_path, mock_get_las, mock_st):
        # Setup session state on the mocked st
        mock_st.session_state = self.session_state
        
        # Setup initial state
        self.session_state['review_data'] = None
        self.session_state['la_options'] = ['Old Auth']
        
        # Setup mocks
        mock_parse_bq_path.return_value = ('proj', 'data', 'table')
        mock_get_las.return_value = ['Auth A', 'Auth B']
        
        # Mock st.columns
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        # Configure st.button to return True for "Refresh Authorities"
        def button_side_effect(label, **kwargs):
            if label == "Refresh Authorities":
                return True
            return False
        
        mock_st.button.side_effect = button_side_effect
        mock_st.sidebar.__enter__.return_value = mock_st.sidebar
        
        # Run the app function
        main_ui()
        
        # Assertions
        mock_get_las.assert_called()
        self.assertEqual(self.session_state['la_options'], ['Auth A', 'Auth B'])
        mock_st.success.assert_called_with("Refreshed list. Found 2 authorities.")

if __name__ == '__main__':
    unittest.main()