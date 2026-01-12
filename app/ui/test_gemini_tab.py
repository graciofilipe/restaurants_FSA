import unittest
from unittest.mock import MagicMock, patch
from app.ui.st_app import main_ui

class TestGeminiTab(unittest.TestCase):
    @patch('app.ui.st_app.st')
    @patch('app.ui.st_app.execute_gemini_enrichment')
    @patch('app.ui.st_app.get_selected_rows')
    @patch('app.ui.st_app.load_filtered_data_from_bq')
    def test_gemini_analysis_button_logic(self, mock_load, mock_get_rows, mock_execute, mock_st):
        """Test Gemini Analysis button enabled/disabled state."""
        # Setup common mocks
        mock_st.session_state = {'review_data': [{'fhrsid': '123'}]}
        mock_st.sidebar = MagicMock()
        mock_st.text_input.return_value = "proj.ds.table"
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        # --- Case 1: With Selection ---
        mock_get_rows.return_value = [{'fhrsid': '123'}]
        # Mock button click to True so we can verify execute is called
        def button_side_effect(label, **kwargs):
            if "Run Gemini Analysis" in label and not kwargs.get('disabled'):
                return True
            return False
        mock_st.button.side_effect = button_side_effect
        
        main_ui()
        
        # Verify execute called
        mock_execute.assert_called()
        args = mock_execute.call_args[1]
        self.assertEqual(args['fhrsids'], ['123'])
        
        # --- Case 2: No Selection ---
        mock_execute.reset_mock()
        mock_st.button.reset_mock()
        mock_get_rows.return_value = []
        
        main_ui()
        
        # Find disabled button call
        disabled_button_found = False
        for call in mock_st.button.call_args_list:
            if "Run Gemini Analysis" in call[0][0] and call[1].get('disabled') is True:
                disabled_button_found = True
                break
        
        self.assertTrue(disabled_button_found, "Should find disabled button when no rows selected")
        mock_execute.assert_not_called()

if __name__ == '__main__':
    unittest.main()
