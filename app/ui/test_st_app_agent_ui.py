import unittest
from unittest.mock import MagicMock, patch
from app.ui.st_app import display_data
from app.ui.agent_research import render_agent_research_tab

class TestStAppAgentUI(unittest.TestCase):
    @patch('app.ui.st_app.st')
    def test_display_data_enables_selection(self, mock_st):
        data = [{"col1": 1}]
        
        display_data(data)
        
        mock_st.dataframe.assert_called_once()
        call_kwargs = mock_st.dataframe.call_args[1]
        
        assert call_kwargs.get("on_select") == "rerun"
        assert call_kwargs.get("selection_mode") == "single-row"

    @patch('app.ui.agent_research.st')
    def test_render_agent_research_tab_logic(self, mock_st):
        """Test the logic of the Agent Research tab."""
        # Setup mocks
        mock_st.session_state = {}
        
        # Case 1: No selection
        render_agent_research_tab("proj", "data", [])
        
        # Verify info message for no selection
        found_info = False
        for call in mock_st.info.call_args_list:
            if "Select rows" in call[0][0]:
                found_info = True
        self.assertTrue(found_info, "Should show info when no rows selected")
        
        # Verify disabled button
        found_disabled = False
        for call in mock_st.button.call_args_list:
            if call[1].get('disabled') is True:
                found_disabled = True
        self.assertTrue(found_disabled, "Button should be disabled when no selection")
        
        # Case 2: With selection
        mock_st.reset_mock()
        selected_rows = [{"fhrsid": "123"}]
        render_agent_research_tab("proj", "data", selected_rows)
        
        # Verify enabled button
        found_enabled = False
        expected_label = f"Generate Agent Insights ({len(selected_rows)} Restaurants)"
        for call in mock_st.button.call_args_list:
            if call[0][0] == expected_label and not call[1].get('disabled', False):
                found_enabled = True
        self.assertTrue(found_enabled, "Button should be enabled with selection")

if __name__ == '__main__':
    unittest.main()