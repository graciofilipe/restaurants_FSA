import json
import re
import logging
from typing import Dict, Any, Optional
import traceback
import vertexai
from vertexai.preview import reasoning_engines

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration for Remote Agent
PROJECT_ID = "filipegracio-ai-learning"
LOCATION = "us-central1"
AGENT_RESOURCE_ID = "projects/257470209980/locations/us-central1/reasoningEngines/8073464293619662848"

def parse_agent_response(response_text: str) -> Dict[str, Any]:
    """
    Parses the raw text response from the agent to extract structured data.
    Expected format: JSON or Markdown code block with JSON.
    Fields: cuisine_type (str), review_count (int), average_rating (float).
    """
    logger.info(f"Parsing agent response (length: {len(response_text) if response_text else 0})")
    default_result = {
        "cuisine_type": None,
        "review_count": None,
        "average_rating": None
    }
    
    if not response_text:
        logger.warning("Agent response is empty.")
        return default_result

    try:
        clean_text = response_text.strip()
        
        # Remove markdown code blocks if strictly wrapping the json
        if clean_text.startswith("```"):
            # Find the first {
            start = clean_text.find("{")
            # Find the last }
            end = clean_text.rfind("}")
            if start != -1 and end != -1:
                 clean_text = clean_text[start:end+1]
        
        # Just try to find the first { and last } regardless
        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            result = {
                "cuisine_type": data.get("cuisine_type"),
                "review_count": data.get("review_count"),
                "average_rating": data.get("average_rating")
            }
            logger.info(f"Successfully parsed data: {result}")
            return result
        else:
            logger.warning(f"No JSON found in agent response. First 100 chars: {response_text[:100]}...")
            return default_result
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode JSON from agent response: {e}. First 100 chars: {response_text[:100]}...")
        return default_result
    except Exception as e:
        logger.error(f"Error parsing agent response: {e}", exc_info=True)
        return default_result

def get_agent_insight(restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calls the remote Vertex AI Agent to get insights for a single restaurant.
    """
    business_name = restaurant.get("businessname")
    logger.info(f"Starting get_agent_insight for: {business_name}")
    
    address = f"{restaurant.get('addressline1', '')}, {restaurant.get('postcode', '')}"
    
    prompt = (
        f"Research the restaurant '{business_name}' located at '{address}'. "
        "Find out the following details:\n"
        "1. Type of restaurant (Cuisine)\n"
        "2. Number of reviews on Google Maps\n"
        "3. Average rating on Google Maps\n\n"
        "Output the result strictly in JSON format with keys: "
        "'cuisine_type', 'review_count' (integer), 'average_rating' (float)."
    )
    
    try:
        logger.info(f"Initializing Vertex AI (Project: {PROJECT_ID}, Location: {LOCATION})...")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        logger.info(f"Connecting to Remote Agent: {AGENT_RESOURCE_ID}...")
        remote_agent = reasoning_engines.ReasoningEngine(AGENT_RESOURCE_ID)
        
        logger.info("Querying remote agent...")
        # Querying the agent. The input argument depends on how the agent was defined.
        # Assuming standard text input.
        response = remote_agent.query(input=prompt)
        
        logger.info(f"Agent query returned. Type: {type(response)}")
        
        # Handle response format. 
        # If it returns a string, use it. If it returns an object, try to convert to string.
        raw_text = str(response)
        
        if hasattr(response, 'text'):
             raw_text = response.text
        elif isinstance(response, dict):
             raw_text = json.dumps(response)
        
        logger.info(f"Raw response text (first 100 chars): {raw_text[:100]}")

        if not raw_text:
             logger.warning(f"Agent returned no text for {business_name}")
             return None

        parsed = parse_agent_response(raw_text)
        parsed["raw_insight"] = raw_text
        parsed["fhrsid"] = restaurant.get("fhrsid")
        
        return parsed
        
    except Exception as e:
        logger.error(f"Error calling agent for {business_name}: {e}")
        logger.error(traceback.format_exc())
        return None