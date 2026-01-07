from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool

# Instantiate Google Maps tool
# Check if it needs any config. Usually generic maps grounding doesn't need much if enabled on project.
google_maps = GoogleMapsGroundingTool()

AGENT_INSTRUCTION = (
    "You are a helpful assistant that answers questions about restaurants. "
    "You have access to Google Maps to find real-world location information. "
    "Always use the Google Maps tool to verify addresses and details."
)

# Define the Agent
root_agent = Agent(
    name="restaurant_maps_agent",
    model="gemini-2.0-flash-exp",
    instruction=AGENT_INSTRUCTION,
    tools=[google_maps]
)

app = App(root_agent=root_agent, name="restaurant_maps_agent")