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
        "average_rating": 4.5,
        "summary": None
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_plain_json():
    response_text = '{"cuisine_type": "Indian", "review_count": 50, "average_rating": 4.0}'
    expected = {
        "cuisine_type": "Indian",
        "review_count": 50,
        "average_rating": 4.0,
        "summary": None
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_with_summary():
    response_text = '{"cuisine_type": "French", "review_count": 10, "average_rating": 4.8, "summary": "Great food."}'
    expected = {
        "cuisine_type": "French",
        "review_count": 10,
        "average_rating": 4.8,
        "summary": "Great food."
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_missing_fields():
    response_text = '{"cuisine_type": "Pub"}'
    expected = {
        "cuisine_type": "Pub",
        "review_count": None,
        "average_rating": None,
        "summary": None
    }
    assert parse_agent_response(response_text) == expected

def test_parse_agent_response_invalid_json():
    response_text = "I found this place but I can't give you JSON."
    expected = {
        "cuisine_type": None,
        "review_count": None,
        "average_rating": None,
        "summary": None # Because default_result now implicitly includes summary? No, let's check default_result in implementation.
    }
    # Wait, need to check if default_result in implementation was updated.
    # Yes, I saw it in my previous read.
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

@patch('app.services.agent_orchestrator.Client')
def test_get_agent_insight_includes_extra_address_fields(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_agent = MagicMock()
    mock_client.agent_engines.get.return_value = mock_agent
    mock_agent.stream_query.return_value = [] 

    restaurant = {
        "businessname": "Expanded Place", 
        "addressline1": "Line 1", 
        "addressline2": "Line 2", 
        "postcode": "SW1", 
        "localauthorityname": "Camden",
        "fhrsid": "999"
    }

    get_agent_insight(restaurant)
    
    # Verify prompt content
    args, kwargs = mock_agent.stream_query.call_args
    prompt = kwargs['message']
    
    assert "Line 2" in prompt
    assert "Camden" in prompt
