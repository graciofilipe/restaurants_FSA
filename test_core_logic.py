import pytest
from unittest.mock import patch, MagicMock
from app.core.data_processing import parse_coordinates, fetch_data_for_all_coordinates

def test_parse_coordinates_valid():
    input_str = "0.1, 51.5\n-0.2, 52.0"
    expected = [(0.1, 51.5), (-0.2, 52.0)]
    result, errors = parse_coordinates(input_str)
    assert result == expected
    assert len(errors) == 0

def test_parse_coordinates_mixed():
    input_str = "0.1, 51.5\ninvalid_line\n-0.2, 52.0"
    expected_valid = [(0.1, 51.5), (-0.2, 52.0)]
    result, errors = parse_coordinates(input_str)
    assert result == expected_valid
    assert len(errors) == 1
    assert "invalid_line" in errors[0]

def test_parse_coordinates_empty():
    result, errors = parse_coordinates("")
    assert result == []
    assert errors == []

@patch('app.core.data_processing.fetch_api_data')
def test_fetch_data_for_all_coordinates(mock_fetch):
    # Setup mock
    # First call returns one establishment, second call returns None (end of pagination/list)
    mock_fetch.side_effect = [
        {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [{'id': 1}]}}},
        None 
    ]
    
    coords = [(0.1, 51.5)]
    results = fetch_data_for_all_coordinates(coords, max_results=10)
    
    assert len(results) == 1
    assert results[0]['id'] == 1
    assert mock_fetch.call_count >= 1
