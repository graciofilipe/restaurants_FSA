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

@patch('app.services.agent_orchestrator.get_auth_token')
@patch('app.services.agent_orchestrator.requests.post')
def test_get_agent_insight_success(mock_post, mock_get_token):
    mock_get_token.return_value = "fake_token"
    
    # Mock response to return SSE data
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    # Create SSE stream data mimicking Vertex AI streamQuery output
    inner_json = json.dumps({
        "cuisine_type": "Italian",
        "review_count": 100,
        "average_rating": 4.5
    })
    
    chunk_obj = {
        "output": {
            "stringValue": inner_json
        }
    }
    chunk_content = json.dumps(chunk_obj)
    
    # iter_lines should yield bytes
    mock_response.iter_lines.return_value = [
        f"data: {chunk_content}".encode('utf-8'),
        b"",
        b"data: [DONE]"
    ]
    
    mock_post.return_value = mock_response

    restaurant = {"businessname": "Test Place", "addressline1": "123 Street", "postcode": "ABC", "fhrsid": "123"}

    result = get_agent_insight(restaurant)
    
    # Our mocked response returns the JSON inside stringValue. 
    # parse_agent_response should handle it.
    
    assert result is not None
    assert result["cuisine_type"] == "Italian"
    assert result["review_count"] == 100
    assert result["average_rating"] == 4.5
    
    # Verify calls
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "streamQuery" in args[0]
    assert kwargs['json']['input']['message']['parts'][0]['text']

@patch('app.services.agent_orchestrator.get_auth_token')
@patch('app.services.agent_orchestrator.requests.post')
def test_get_agent_insight_failure(mock_post, mock_get_token):
    mock_get_token.return_value = "fake_token"
    
    # Mock response failure
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    mock_post.return_value = mock_response
    
    restaurant = {"businessname": "Test Place", "fhrsid": "123"}
    result = get_agent_insight(restaurant)
    
    assert result is None