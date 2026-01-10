import sys
from unittest.mock import MagicMock, patch, call
import pytest

# Mocking streamlit before import
mock_st = MagicMock()
mock_st.session_state = {}
sys.modules['streamlit'] = mock_st
sys.modules['streamlit.components.v1'] = MagicMock()

# Import the module to test
# We will need to import handle_insight_generation from app.ui.st_app
# It doesn't exist yet, so this import will fail, which is part of the "Red" phase.
try:
    from app.ui.st_app import handle_insight_generation
except ImportError:
    pass

@patch('app.ui.st_app.get_agent_insight')
@patch('app.ui.st_app.upsert_agent_insight')
def test_handle_insight_generation_updates_state(mock_upsert, mock_get_insight):
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
    # Note: handle_insight_generation is expected to take: (targets, project_id, dataset_id, progress_bar, status_text)
    # and return (success_count, total)
    success_count, total = handle_insight_generation(targets, project_id, dataset_id, mock_progress_bar, mock_status_text)
    
    # Assertions
    assert success_count == 1
    assert total == 2
    
    # Verify Session State Logic
    # It should store the SUCCESSFUL fhrsids in session_state['latest_batch_fhrsids']
    assert 'latest_batch_fhrsids' in mock_st.session_state
    assert mock_st.session_state['latest_batch_fhrsids'] == ['1']
    
    # It should NOT contain '2' because agent failed
    
    # It should set a flag to show results
    assert 'show_latest_insights' in mock_st.session_state
    assert mock_st.session_state['show_latest_insights'] is True

@patch('app.ui.st_app.get_agent_insight')
@patch('app.ui.st_app.upsert_agent_insight')
def test_handle_insight_generation_clears_previous_state(mock_upsert, mock_get_insight):
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
    
    assert mock_st.session_state['latest_batch_fhrsids'] == ['new_id']
    assert mock_st.session_state['show_latest_insights'] is True

@patch('app.ui.st_app.load_specific_agent_insights')
@patch('app.ui.st_app.parse_bq_path')
def test_ui_displays_latest_insights(mock_parse_bq, mock_load):
    # Setup state
    mock_st.session_state = {
        'review_data': [{'fhrsid': '1', 'businessname': 'B1'}],
        'show_latest_insights': True,
        'latest_batch_fhrsids': ['1'],
        'la_options': [] 
    }
    
    # Mock return values
    mock_parse_bq.return_value = ("p", "d", "t")
    mock_load.return_value = [{'fhrsid': '1', 'insight': 'data'}]
    
    # Configure interactive widgets to avoid entering unexpected blocks
    mock_st.button.return_value = False
    mock_st.radio.return_value = "Batch (All Filtered)" # Default to something valid
    
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
    mock_st.expander.return_value = mock_expander
    
    mock_spinner = MagicMock()
    mock_spinner.__enter__.return_value = None
    mock_spinner.__exit__.return_value = None
    mock_st.spinner.return_value = mock_spinner

    # Import main_ui
    from app.ui.st_app import main_ui
    
    # Execute
    main_ui()
    
    # Assert
    # Verify load_specific_agent_insights was called with the IDs from session state
    mock_load.assert_called_with("p", "d", ['1'])
    
    # Verify a dataframe was displayed with the loaded data
    # st.dataframe is called for the main list, and then for the results.
    # We check if one of the calls involved our mocked return value.
    # mock_load.return_value is a list, pd.DataFrame(list) creates a DF.
    # Since we mocked pd in st_app (no we didn't mock pd in st_app, we mocked st).
    # Real pandas is used.
    
    # We can check if st.dataframe was called.
    assert mock_st.dataframe.called

