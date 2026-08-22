from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Initializing app/maps_agent/agent.py")

try:
    # Instantiate Google Maps tool
    # Check if it needs any config. Usually generic maps grounding doesn't need much if enabled on project.
    logger.info("Instantiating GoogleMapsGroundingTool...")
    google_maps = GoogleMapsGroundingTool()
    logger.info(f"GoogleMapsGroundingTool instantiated: {type(google_maps)}")

    AGENT_INSTRUCTION = (
        "You are a helpful assistant that answers questions about restaurants. "
        "You have access to Google Maps to find real-world location information. "
        "Always use the Google Maps tool to verify addresses and details. "
        "You MUST output your response strictly in JSON format. "
        "The JSON should contain keys like 'cuisine_type', 'review_count', 'average_rating', and 'summary'."
    )

    # Define the Agent
    logger.info("Instantiating Agent...")
    root_agent = Agent(
        name="restaurant_maps_agent",
        model="gemini-3.7-flash",
        instruction=AGENT_INSTRUCTION,
        tools=[google_maps]
    )
    logger.info(f"Agent instantiated: {type(root_agent)}")
    logger.info(f"Agent attributes: {dir(root_agent)}")

    app = App(root_agent=root_agent, name="maps_agent")
    logger.info("App instantiated.")

except Exception as e:
    logger.error(f"CRITICAL ERROR in app/maps_agent/agent.py: {e}", exc_info=True)
    raise e