from google import genai
from google.genai import types
import os

class RestaurantAgent:
    """An agent that answers questions about restaurants using Google Maps Grounding."""
    
    def __init__(self, project_id: str, location: str = "us-central1", model_name: str = "gemini-2.0-flash-exp"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

    def query(self, prompt: str) -> str:
        """
        Queries the agent with a prompt.
        
        Args:
            prompt: The user's question.
            
        Returns:
            The agent's response text.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_maps=types.GoogleMaps())],
            )
        )
        return response.text
