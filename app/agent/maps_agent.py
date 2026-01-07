from google.adk.agents import LlmAgent
from google.adk.tools import google_maps_grounding

def create_restaurant_agent(project_id: str, location: str = "us-central1", model_name: str = "gemini-2.0-flash-exp") -> LlmAgent:
    """Creates a Restaurant Agent using Google ADK."""
    
    agent = LlmAgent(
        name="restaurant_maps_agent",
        model=model_name,
        instruction="You are a helpful assistant that answers questions about restaurants. "
                    "You have access to Google Maps to find real-world location information. "
                    "Always use the Google Maps tool to verify addresses and details.",
        tools=[google_maps_grounding]
    )
    return agent
