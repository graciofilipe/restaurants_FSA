import pytest
from unittest.mock import patch, MagicMock
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

@patch('app.services.agent_orchestrator.vertexai.init')
@patch('app.services.agent_orchestrator.reasoning_engines.ReasoningEngine')
def test_get_agent_insight_success(mock_engine_cls, mock_init):
    # Setup mock engine
    mock_agent = mock_engine_cls.return_value
    
    # Setup mock response object
    mock_response = MagicMock()
    # If the response is an object with a text attribute
    mock_response.text = '{"cuisine_type": "Italian", "review_count": 100, "average_rating": 4.5}'
    # Or if it behaves like a dict
    mock_response.get.side_effect = lambda k: {"cuisine_type": "Italian"}.get(k)
    
    # We configure query to return this object
    mock_agent.query.return_value = mock_response
    
    restaurant = {"businessname": "Test Place", "addressline1": "123 Street", "postcode": "ABC", "fhrsid": "123"}
    
    result = get_agent_insight(restaurant)
    
    assert result["cuisine_type"] == "Italian"
    assert result["raw_insight"] == mock_response.text
    
    # Verify instantiation and call
    mock_init.assert_called_once()
    mock_engine_cls.assert_called_once()
    mock_agent.query.assert_called_once()
    args, kwargs = mock_agent.query.call_args
    assert "Test Place" in kwargs['input']

@patch('app.services.agent_orchestrator.vertexai.init')
@patch('app.services.agent_orchestrator.reasoning_engines.ReasoningEngine')
def test_get_agent_insight_failure(mock_engine_cls, mock_init):
    mock_agent = mock_engine_cls.return_value
    mock_agent.query.side_effect = Exception("Agent Error")
    
    restaurant = {"businessname": "Test Place", "fhrsid": "123"}
    result = get_agent_insight(restaurant)
    
    assert result is None