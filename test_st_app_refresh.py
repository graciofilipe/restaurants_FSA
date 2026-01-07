
import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock streamlit before importing app modules if they import streamlit at top level
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Now import the app module
# We need to ensure bq_utils is also mocked or available
from app.ui.st_app import main_ui

class SessionState(dict):
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

class TestStAppRefresh(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        st.reset_mock()
        # Setup session state mock
        self.session_state = SessionState()
        st.session_state = self.session_state

    @patch('app.ui.st_app.get_distinct_local_authorities')
    @patch('app.ui.st_app.parse_bq_path')
    def test_refresh_authorities_button(self, mock_parse_bq_path, mock_get_las):
        # Setup initial state
        self.session_state['review_data'] = None
        # Pre-populate la_options to simulate existing state
        self.session_state['la_options'] = ['Old Auth']
        
        # Setup mocks
        mock_parse_bq_path.return_value = ('proj', 'data', 'table')
        mock_get_las.return_value = ['Auth A', 'Auth B']
        
        # Configure st.button to return True for "Refresh Authorities"
        # The app calls st.button("Refresh Authorities") inside the sidebar
        # We need to capture the sidebar context or just mock button return values based on label
        
        def button_side_effect(label, **kwargs):
            if label == "Refresh Authorities":
                return True
            return False
        
        st.button.side_effect = button_side_effect
        st.sidebar.__enter__ = MagicMock(return_value=st.sidebar)
        st.sidebar.__exit__ = MagicMock(return_value=None)
        
        # Run the app function
        main_ui()
        
        # Assertions
        # 1. Check if get_distinct_local_authorities was called
        mock_get_las.assert_called()
        
        # 2. Check if session state was updated
        self.assertEqual(self.session_state['la_options'], ['Auth A', 'Auth B'])
        
        # 3. Check for success message
        st.success.assert_called_with("Refreshed list. Found 2 authorities.")

if __name__ == '__main__':
    unittest.main()
