import pytest
from data_processing import load_master_data, process_and_update_master_data
from unittest.mock import MagicMock, patch

# Test cases for load_master_data
def test_load_master_data_initializes_manual_review():
    """
    Tests that load_master_data initializes 'manual_review' to 'not reviewed'
    if it's missing from a record.
    """
    mock_load_json_func = MagicMock()
    input_data = [
        {"FHRSID": "1", "BusinessName": "Restaurant A"}, # Missing manual_review
        {"FHRSID": "2", "BusinessName": "Restaurant B", "manual_review": "reviewed"}, # Has manual_review
        {"FHRSID": "3", "BusinessName": "Restaurant C", "gemini_insights": "has_insight"}, # Missing manual_review, gemini_insights should be preserved
        {"FHRSID": "4", "BusinessName": "Restaurant D", "manual_review": "reviewed", "gemini_insights": "has_insight"} # Has both, gemini_insights should be preserved
    ]
    mock_load_json_func.return_value = input_data

    expected_data = [
        {"FHRSID": "1", "BusinessName": "Restaurant A", "manual_review": "not reviewed"},
        {"FHRSID": "2", "BusinessName": "Restaurant B", "manual_review": "reviewed"},
        {"FHRSID": "3", "BusinessName": "Restaurant C", "manual_review": "not reviewed", "gemini_insights": "has_insight"},
        {"FHRSID": "4", "BusinessName": "Restaurant D", "manual_review": "reviewed", "gemini_insights": "has_insight"}
    ]

    result = load_master_data("dummy_uri", mock_load_json_func)
    assert result == expected_data, "manual_review not initialized correctly"

def test_load_master_data_empty_input():
    mock_load_json_func = MagicMock(return_value=[])
    result = load_master_data("dummy_uri", mock_load_json_func)
    assert result == []

def test_load_master_data_none_input():
    mock_load_json_func = MagicMock(return_value=None)
    result = load_master_data("dummy_uri", mock_load_json_func)
    assert result == []

# Test cases for process_and_update_master_data
def test_process_and_update_master_data_initializes_fields_for_new_records():
    """
    Tests that process_and_update_master_data initializes 'manual_review'
    and 'first_seen' for new records.
    """
    master_data = [
        {"FHRSID": "1", "BusinessName": "Existing Restaurant", "manual_review": "reviewed"}
    ]

    # Ensure 'first_seen' is also handled as the function expects it
    api_data = {
        "FHRSEstablishment": {
            "EstablishmentCollection": {
                "EstablishmentDetail": [
                    {"FHRSID": "2", "BusinessName": "New Restaurant 1"}, # New, will be added
                    {"FHRSID": "1", "BusinessName": "Existing Restaurant Updated Info"} # Existing, will be skipped for adding
                ]
            }
        }
    }

    # process_and_update_master_data modifies master_data in place
    # and also returns it. It also returns the count of new restaurants.
    # It also adds 'first_seen'. We'll mock datetime for predictable 'first_seen'.

    with patch('data_processing.datetime') as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01" # Mock 'first_seen' date

        updated_master_data, new_restaurants_count = process_and_update_master_data(master_data, api_data)

        assert new_restaurants_count == 1
        assert len(updated_master_data) == 2

        # Check existing restaurant (should be unchanged by this aspect of the function)
        assert updated_master_data[0]["FHRSID"] == "1"
        # gemini_insights was removed from existing_restaurant data, so no check needed here for it
        assert updated_master_data[0]["manual_review"] == "reviewed"

        # Check new restaurant
        new_restaurant = next(r for r in updated_master_data if r["FHRSID"] == "2")
        assert new_restaurant["BusinessName"] == "New Restaurant 1"
        assert new_restaurant["manual_review"] == "not reviewed"
        assert new_restaurant["first_seen"] == "2024-01-01"

def test_process_and_update_master_data_no_new_records():
    master_data = [{"FHRSID": "1", "BusinessName": "Restaurant A", "manual_review": "reviewed"}] # Removed gemini_insights
    api_data = {
        "FHRSEstablishment": {
            "EstablishmentCollection": {
                "EstablishmentDetail": [
                    {"FHRSID": "1", "BusinessName": "Restaurant A Updated"}
                ]
            }
        }
    }
    updated_data, new_count = process_and_update_master_data(list(master_data), api_data) # Pass a copy
    assert new_count == 0
    assert len(updated_data) == 1
    # No assertion for gemini_insights needed here as it's removed from master_data

def test_process_and_update_master_data_empty_api_details():
    master_data = [{"FHRSID": "1", "BusinessName": "Restaurant A"}]
    api_data = {
        "FHRSEstablishment": {
            "EstablishmentCollection": {
                "EstablishmentDetail": [] # No new establishments
            }
        }
    }
    updated_data, new_count = process_and_update_master_data(list(master_data), api_data)
    assert new_count == 0
    assert len(updated_data) == 1

def test_process_and_update_master_data_malformed_api_data_no_details():
    master_data = [{"FHRSID": "1", "BusinessName": "Restaurant A"}]
    api_data = { # Missing EstablishmentDetail
        "FHRSEstablishment": {
            "EstablishmentCollection": {}
        }
    }
    updated_data, new_count = process_and_update_master_data(list(master_data), api_data)
    assert new_count == 0
    assert len(updated_data) == 1

def test_process_and_update_master_data_malformed_api_data_no_collection():
    master_data = [{"FHRSID": "1", "BusinessName": "Restaurant A"}]
    api_data = { # Missing EstablishmentCollection
        "FHRSEstablishment": {}
    }
    updated_data, new_count = process_and_update_master_data(list(master_data), api_data)
    assert new_count == 0
    assert len(updated_data) == 1

def test_process_and_update_master_data_malformed_api_data_no_fhrse():
    master_data = [{"FHRSID": "1", "BusinessName": "Restaurant A"}]
    api_data = {} # Missing FHRSEstablishment
    updated_data, new_count = process_and_update_master_data(list(master_data), api_data)
    assert new_count == 0
    assert len(updated_data) == 1
