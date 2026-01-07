import pytest
from app.agent.maps_agent import create_restaurant_agent
from google.adk.agents import LlmAgent

from google.adk.tools import google_maps_grounding

def test_create_restaurant_agent():
    agent = create_restaurant_agent(project_id="test-project")
    
    assert isinstance(agent, LlmAgent)
    assert agent.name == "restaurant_maps_agent"
    assert agent.model == "gemini-2.0-flash-exp"
    
    # Check tools
    assert agent.tools is not None
    assert len(agent.tools) == 1
    assert agent.tools[0] == google_maps_grounding

