import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.ui.st_app import display_data, get_selected_rows

def test_display_data_call():
    df = pd.DataFrame({'col1': [1, 2]})
    with patch('streamlit.dataframe') as mock_df:
        display_data(df)
        mock_df.assert_called_once()

def test_get_selected_rows():
    mock_event = MagicMock()
    mock_event.selection.rows = [0]
    df = pd.DataFrame({'col1': [10, 20]})
    
    result = get_selected_rows(mock_event, df)
    assert len(result) == 1
    assert result.iloc[0]['col1'] == 10
