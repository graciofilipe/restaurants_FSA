import pandas as pd
from unittest.mock import MagicMock, patch
from app.ui.st_app import get_selected_rows, display_data

class MockSelection:
    def __init__(self, rows):
        self.rows = rows

class MockEvent:
    def __init__(self, selection_rows=None):
        if selection_rows is not None:
            self.selection = MockSelection(selection_rows)
        else:
            self.selection = None

def test_get_selected_rows_basic():
    df = pd.DataFrame([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
    # Event with row 0 selected
    event = MockEvent(selection_rows=[0])
    
    selected = get_selected_rows(event, df)
    
    assert selected is not None
    assert len(selected) == 1
    assert selected.iloc[0]["name"] == "A"

def test_get_selected_rows_second_item():
    df = pd.DataFrame([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
    # Event with row 1 selected
    event = MockEvent(selection_rows=[1])
    
    selected = get_selected_rows(event, df)
    
    assert selected is not None
    assert len(selected) == 1
    assert selected.iloc[0]["name"] == "B"

def test_get_selected_rows_empty_selection():
    df = pd.DataFrame([{"id": 1, "name": "A"}])
    event = MockEvent(selection_rows=[])
    
    selected = get_selected_rows(event, df)
    
    # Implementation returns None if no selection
    assert selected is None

def test_get_selected_rows_none_event():
    df = pd.DataFrame([{"id": 1, "name": "A"}])
    event = None
    
    selected = get_selected_rows(event, df)
    
    assert selected is None


@patch('app.ui.st_app.st')
def test_display_data_enables_selection(mock_st):
    data = [{"col1": 1}]
    
    display_data(data)
    
    mock_st.dataframe.assert_called_once()
    call_kwargs = mock_st.dataframe.call_args[1]
    
    assert call_kwargs.get("on_select") == "rerun"
    assert call_kwargs.get("selection_mode") == "multi-row"
