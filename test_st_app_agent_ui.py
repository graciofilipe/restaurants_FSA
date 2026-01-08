import sys
from unittest.mock import MagicMock, patch
import pandas as pd

# Mocking streamlit
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st
sys.modules['streamlit.components.v1'] = MagicMock()

# Import after mocking
# We need to mock other dependencies that might be imported at top level if they have side effects
# app.services.bq_utils and app.core.data_processing should be safe to import if dependencies are met.
# But let's mock them just in case to be safe and isolate UI testing.
sys.modules['app.services.bq_utils'] = MagicMock()
sys.modules['app.core.data_processing'] = MagicMock()
sys.modules['app.services.agent_orchestrator'] = MagicMock()

# Now import st_app
try:
    from app.ui.st_app import display_data
except ImportError:
    # If imports fail due to mocked modules not having attributes used in global scope (unlikely here)
    pass

def test_display_data_enables_selection():
    mock_st.dataframe.reset_mock()
    data = [{"col1": 1}]
    
    # Reload module or ensure display_data uses the mocked st
    from app.ui.st_app import display_data
    
    display_data(data)
    
    mock_st.dataframe.assert_called_once()
    call_kwargs = mock_st.dataframe.call_args[1]
    
    assert call_kwargs.get("on_select") == "rerun"
    assert call_kwargs.get("selection_mode") == "multi-row"

def test_main_ui_has_agent_tab_logic():
    # Verify the file content contains the logic we added.
    with open('app/ui/st_app.py', 'r') as f:
        content = f.read()
    
    assert 'tab_gemini, tab_agent = st.tabs' in content
    assert 'Generate Agent Insights' in content
    assert 'get_agent_insight(restaurant)' in content
