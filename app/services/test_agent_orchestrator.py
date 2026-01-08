import pytest
from unittest.mock import patch, MagicMock
try:
    from app.services.agent_orchestrator import parse_agent_response, get_agent_insight
except ImportError:
    # Allow partial import if function doesn't exist yet
    try:
        from app.services.agent_orchestrator import parse_agent_response
    except ImportError:
        pass

def test_parse_agent_response_json():
    response_text = """```json
{"cuisine_type": "Italian", "review_count": 100, "average_rating": 4.5}
```"""
    expected = {
        "cuisine_type": "Italian",
        "review_count": 100,
        "average_rating": 4.5
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_plain_json():
    response_text = '{"cuisine_type": "Indian", "review_count": 50, "average_rating": 4.0}'
    expected = {
        "cuisine_type": "Indian",
        "review_count": 50,
        "average_rating": 4.0
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_missing_fields():
    response_text = '{"cuisine_type": "Pub"}'
    expected = {
        "cuisine_type": "Pub",
        "review_count": None,
        "average_rating": None
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_invalid_json():
    response_text = "I found this place but I can't give you JSON."
    expected = {
        "cuisine_type": None,
        "review_count": None,
        "average_rating": None
    }
    assert parse_agent_response(response_text) == expected

@patch('app.services.agent_orchestrator.root_agent')
def test_get_agent_insight_success(mock_agent):
    # Setup mock
    mock_response = MagicMock()
    mock_response.text = '{"cuisine_type": "Italian", "review_count": 100, "average_rating": 4.5}'
    mock_agent.chat.return_value = mock_response
    
    restaurant = {"businessname": "Test Place", "addressline1": "123 Street", "postcode": "ABC"}
    
    result = get_agent_insight(restaurant)
    
    assert result["cuisine_type"] == "Italian"
    assert result["raw_insight"] == mock_response.text
    assert "Test Place" in mock_agent.chat.call_args[0][0] # prompt should contain name

@patch('app.services.agent_orchestrator.root_agent')
def test_get_agent_insight_failure(mock_agent):
    mock_agent.chat.side_effect = Exception("Agent Error")
    
    restaurant = {"businessname": "Test Place"}
    result = get_agent_insight(restaurant)
    
    assert result is None
