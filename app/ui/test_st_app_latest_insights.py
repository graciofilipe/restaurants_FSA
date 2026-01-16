import unittest
from unittest.mock import MagicMock, patch, call
import pytest
from app.ui.agent_research import handle_insight_generation
from app.ui.st_app import main_ui

class TestStAppLatestInsights(unittest.TestCase):

    @patch('app.ui.agent_research.st')
    @patch('app.ui.agent_research.get_agent_insight')
    @patch('app.ui.agent_research.upsert_agent_insight')
    def test_handle_insight_generation_updates_state(self, mock_upsert, mock_get_insight, mock_st):
        # Setup
        mock_st.session_state = {}
        mock_progress_bar = MagicMock()
        mock_status_text = MagicMock()
        
        # Mock Data
        targets = [
            {'fhrsid': '1', 'businessname': 'Rest 1'},
            {'fhrsid': '2', 'businessname': 'Rest 2'}
        ]
        project_id = "test-proj"
        dataset_id = "test-data"
        
        # Mock Agent Responses
        # Rest 1: Success
        mock_get_insight.side_effect = [
            {'fhrsid': '1', 'raw_insight': 'Insight 1'},
            None # Rest 2: Failure (agent returns None)
        ]
        
        # Mock Upsert (Success for Rest 1)
        mock_upsert.return_value = True
        
        # Execution
        success_count, total = handle_insight_generation(targets, project_id, dataset_id, mock_progress_bar, mock_status_text)
        
        # Assertions
        self.assertEqual(success_count, 1)
        self.assertEqual(total, 2)
        
        # Verify Session State Logic
        self.assertIn('latest_batch_fhrsids', mock_st.session_state)
        self.assertEqual(mock_st.session_state['latest_batch_fhrsids'], ['1'])
        self.assertTrue(mock_st.session_state['show_latest_insights'])

    @patch('app.ui.agent_research.st')
    @patch('app.ui.agent_research.get_agent_insight')
    @patch('app.ui.agent_research.upsert_agent_insight')
    def test_handle_insight_generation_clears_previous_state(self, mock_upsert, mock_get_insight, mock_st):
        # Setup pre-existing state
        mock_st.session_state = {
            'latest_batch_fhrsids': ['old_id'],
            'show_latest_insights': False
        }
        
        mock_progress_bar = MagicMock()
        mock_status_text = MagicMock()
        
        targets = [{'fhrsid': 'new_id', 'businessname': 'New Rest'}]
        
        mock_get_insight.return_value = {'fhrsid': 'new_id', 'raw_insight': 'New Insight'}
        mock_upsert.return_value = True
        
        handle_insight_generation(targets, "p", "d", mock_progress_bar, mock_status_text)
        
        self.assertEqual(mock_st.session_state['latest_batch_fhrsids'], ['new_id'])
        self.assertTrue(mock_st.session_state['show_latest_insights'])

    @patch('app.ui.st_app.st')
    @patch('app.ui.agent_research.st') # Need to patch st in agent_research too as it's called
    @patch('app.ui.agent_research.load_specific_agent_insights')
    @patch('app.ui.st_app.parse_bq_path')
    @patch('app.ui.st_app.enhance_dataframe_with_insights')
    @patch('app.ui.st_app.get_distinct_outcodes')
    @patch('app.ui.st_app.get_distinct_local_authorities')
    @patch('app.ui.st_app.load_filtered_data_from_bq')
    def test_ui_displays_latest_insights(self, mock_load_filtered, mock_get_las, mock_get_outcodes, mock_enhance, mock_parse_bq, mock_load, mock_st_agent, mock_st):
        # Setup state
        # Both mocks need access to session state
        state = {
            'review_data': [{'fhrsid': '1', 'businessname': 'B1'}],
            'show_latest_insights': True,
            'latest_batch_fhrsids': ['1'],
            'la_options': []
        } 
        mock_st.session_state = state
        mock_st_agent.session_state = state
        
        # Mock return values
        mock_parse_bq.return_value = ("p", "d", "t")
        mock_load.return_value = [{'fhrsid': '1', 'insight': 'data'}]
        
        # Configure interactive widgets to avoid entering unexpected blocks
        mock_st.button.return_value = False
        mock_st.button.return_value = False
        mock_st_agent.button.return_value = False
        mock_st_agent.radio.return_value = "Batch (All Filtered)" # Default to something valid
        
        mock_st.slider.return_value = 0
        mock_st.multiselect.return_value = []
        
        mock_get_las.return_value = []
        mock_get_outcodes.return_value = []
        mock_enhance.side_effect = lambda df: df
        mock_load_filtered.return_value = []
        
        # Mock context managers
        mock_col = MagicMock()
        mock_col.__enter__.return_value = mock_col
        mock_col.__exit__.return_value = None
        mock_st.columns.return_value = [mock_col, mock_col]
        
        mock_tab = MagicMock()
        mock_tab.__enter__.return_value = mock_tab
        mock_tab.__exit__.return_value = None
        mock_st.tabs.return_value = [mock_tab, mock_tab]
        
        mock_expander = MagicMock()
        mock_expander.__enter__.return_value = mock_expander
        mock_expander.__exit__.return_value = None
        mock_st.expander.return_value = mock_expander # Or mock_st_agent.expander?
        mock_st_agent.expander.return_value = mock_expander
        
        mock_spinner = MagicMock()
        mock_spinner.__enter__.return_value = None
        mock_spinner.__exit__.return_value = None
        mock_st.spinner.return_value = mock_spinner
        mock_st_agent.spinner.return_value = mock_spinner

        # Execute
        main_ui()
        
        # Assert
        mock_load.assert_called_with("p", "d", ['1'])
        # dataframe display is called on st inside render_agent_research_tab (which uses agent_research.st)
        self.assertTrue(mock_st_agent.dataframe.called)

if __name__ == '__main__':
    unittest.main()
