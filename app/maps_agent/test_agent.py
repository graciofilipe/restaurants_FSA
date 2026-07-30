from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool
from app.maps_agent.agent import root_agent
from google.adk.agents import Agent

def test_restaurant_agent_config():
    agent = root_agent
    
    assert isinstance(agent, Agent)
    assert agent.name == "restaurant_maps_agent"
    assert agent.model == "gemini-3.5-flash"
    
    # Check tools
    assert agent.tools is not None
    assert len(agent.tools) == 1
    # Check type
    assert isinstance(agent.tools[0], GoogleMapsGroundingTool)

