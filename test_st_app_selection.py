import pandas as pd
from app.ui.st_app import get_selected_rows

def test_get_selected_rows_legacy_structure():
    data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    # Legacy/Direct dictionary structure: {'rows': [0]}
    event = {'rows': [0]}
    
    selected = get_selected_rows(event, data)
    
    assert len(selected) == 1
    assert selected[0]["name"] == "A"

def test_get_selected_rows_new_structure():
    data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    # New structure: {'selection': {'rows': [1], ...}}
    event = {'selection': {'rows': [1], 'columns': []}}
    
    selected = get_selected_rows(event, data)
    
    assert len(selected) == 1
    assert selected[0]["name"] == "B"

def test_get_selected_rows_empty_selection():
    data = [{"id": 1, "name": "A"}]
    event = {'selection': {'rows': [], 'columns': []}}
    
    selected = get_selected_rows(event, data)
    
    assert len(selected) == 0

def test_get_selected_rows_none_event():
    data = [{"id": 1, "name": "A"}]
    event = None
    
    selected = get_selected_rows(event, data)
    
    assert len(selected) == 0
