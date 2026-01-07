import pytest
from unittest.mock import MagicMock, patch
from app.agent.maps_agent import RestaurantAgent

@patch("app.agent.maps_agent.genai.Client")
def test_agent_query(mock_client_cls):
    # Setup mock
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = "Nando's is at 123 Chicken St."
    mock_client.models.generate_content.return_value = mock_response

    # Initialize Agent
    agent = RestaurantAgent(project_id="test-project")
    
    # Query
    response = agent.query("Where is Nando's?")
    
    # Verify
    assert response == "Nando's is at 123 Chicken St."
    mock_client.models.generate_content.assert_called_once()
    
    # Verify tools configuration
    call_args = mock_client.models.generate_content.call_args
    config = call_args.kwargs["config"]
    # Check if tools are present
    assert config.tools is not None
    assert len(config.tools) > 0
    assert config.tools[0].google_maps is not None
