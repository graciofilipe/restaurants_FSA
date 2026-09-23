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

import app.core.data_processing

@patch.object(app.core.data_processing, 'fetch_api_data')
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


@patch.object(app.core.data_processing, 'time')
@patch.object(app.core.data_processing, 'fetch_api_data')
def test_fetch_data_for_all_coordinates_stops_at_the_page_cap(mock_fetch, _mock_time):
    """An API that never returns a short page must not loop forever.

    The only exits were a falsy response and a page shorter than max_results,
    so an API that ignores the page parameter would page until the Cloud Run
    job died -- sleeping a second and growing the list each time.
    """
    full_page = {'FHRSEstablishment': {'EstablishmentCollection': {
        'EstablishmentDetail': [{'id': 1}, {'id': 2}]}}}
    mock_fetch.return_value = full_page

    results = fetch_data_for_all_coordinates([(0.1, 51.5)], max_results=2, max_pages=3)

    assert mock_fetch.call_count == 3
    assert len(results) == 6


@patch.object(app.core.data_processing, 'time')
@patch.object(app.core.data_processing, 'fetch_api_data')
def test_fetch_data_for_all_coordinates_warns_when_capped(mock_fetch, _mock_time, caplog):
    mock_fetch.return_value = {'FHRSEstablishment': {'EstablishmentCollection': {
        'EstablishmentDetail': [{'id': 1}]}}}

    fetch_data_for_all_coordinates([(0.1, 51.5)], max_results=1, max_pages=2)

    assert any('page limit' in r.message.lower() for r in caplog.records)


@patch.object(app.core.data_processing, 'time')
@patch.object(app.core.data_processing, 'fetch_api_data')
def test_page_cap_applies_per_coordinate(mock_fetch, _mock_time):
    """One exhausted coordinate must not shorten the next one's fetch."""
    mock_fetch.return_value = {'FHRSEstablishment': {'EstablishmentCollection': {
        'EstablishmentDetail': [{'id': 1}]}}}

    fetch_data_for_all_coordinates([(0.1, 51.5), (0.2, 52.5)], max_results=1, max_pages=2)

    assert mock_fetch.call_count == 4
