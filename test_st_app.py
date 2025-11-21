import unittest
import re
from unittest.mock import patch, MagicMock, PropertyMock, call, ANY
import pandas as pd

from st_app import main_ui

@patch('st_app.st', autospec=True)
class TestMainUIOnly(unittest.TestCase):
    def test_main_ui_displays_fetch_data_section(self, mock_st_global):
        mock_st_global.session_state = MagicMock()
        initial_session_state_attrs = {
            'recent_restaurants_df': None, 'current_project_id': None,
            'current_dataset_id': None, 'displaying_genai_temp': False
        }
        mock_st_global.session_state.configure_mock(**initial_session_state_attrs)

        def session_state_get(name, default=None):
            return getattr(mock_st_global.session_state, name, default)

        def session_state_contains(name):
            return hasattr(mock_st_global.session_state, name)

        mock_st_global.session_state.get = MagicMock(side_effect=session_state_get)
        type(mock_st_global.session_state).__contains__ = MagicMock(side_effect=session_state_contains)

        # Configure st.columns to return correct number of mocks based on input
        def columns_side_effect(num_cols):
            return [MagicMock() for _ in range(num_cols)]
        
        mock_st_global.columns.side_effect = columns_side_effect

        main_ui()

        # Verify the "Fetch API Data" subheader was displayed
        mock_st_global.subheader.assert_any_call("Fetch API Data and Update Master List")
        # Verify the "Gemini Intelligence Analysis" subheader was displayed
        mock_st_global.subheader.assert_any_call("Gemini Intelligence Analysis")
        # Verify the "Export Filtered Data" subheader was displayed
        mock_st_global.subheader.assert_any_call("Export Filtered Data")
        # Verify the "Bulk Update Manual Reviews" subheader was displayed
        mock_st_global.subheader.assert_any_call("Bulk Update Manual Reviews")
        
        mock_st_global.radio.assert_not_called()
