import pytest
from unittest.mock import patch, MagicMock
import json
try:
    from app.services.agent_orchestrator import parse_agent_response, get_agent_insight
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

@patch('app.services.agent_orchestrator.Client')
def test_get_agent_insight_success(mock_client_cls):
    # Setup mock client and agent
    mock_client = mock_client_cls.return_value
    mock_agent = MagicMock()
    mock_client.agent_engines.get.return_value = mock_agent
    
    # Mock stream_query response
    inner_json = json.dumps({
        "cuisine_type": "Italian",
        "review_count": 100,
        "average_rating": 4.5
    })
    
    # SDK returns chunks as dicts
    mock_chunk = {
        "content": {
            "parts": [{"text": inner_json}]
        }
    }
    mock_agent.stream_query.return_value = [mock_chunk]

    restaurant = {"businessname": "Test Place", "addressline1": "123 Street", "postcode": "ABC", "fhrsid": "123"}

    result = get_agent_insight(restaurant)
    
    assert result is not None
    assert result["cuisine_type"] == "Italian"
    assert result["review_count"] == 100
    assert result["average_rating"] == 4.5
    
    # Verify calls
    mock_client_cls.assert_called_once()
    mock_client.agent_engines.get.assert_called_once()
    mock_agent.stream_query.assert_called_once()
    args, kwargs = mock_agent.stream_query.call_args
    assert "Test Place" in kwargs['message']
    assert kwargs['user_id'] == "fsa_reviewer_app"

@patch('app.services.agent_orchestrator.Client')
def test_get_agent_insight_failure(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_agent = MagicMock()
    mock_client.agent_engines.get.return_value = mock_agent
    
    # Mock exception in stream_query
    mock_agent.stream_query.side_effect = Exception("Agent Error")
    
    restaurant = {"businessname": "Test Place", "fhrsid": "123"}
    result = get_agent_insight(restaurant)
    
    assert result is None
