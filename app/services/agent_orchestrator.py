import json
import re
import logging
from typing import Dict, Any, Optional
import pkg_resources
import traceback
from google.adk.runners import InMemoryRunner
from app.maps_agent.agent import root_agent

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log_system_info():
    try:
        logger.info("=== SYSTEM INFO START ===")
        try:
            adk_version = pkg_resources.get_distribution("google-adk").version
            logger.info(f"google-adk version: {adk_version}")
        except Exception as e:
            logger.error(f"Could not determine google-adk version: {e}")
        
        logger.info(f"root_agent type: {type(root_agent)}")
        logger.info(f"root_agent dir: {dir(root_agent)}")
        logger.info("=== SYSTEM INFO END ===")
    except Exception as e:
        logger.error(f"Error logging system info: {e}")

# Log info on module load
log_system_info()

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
        logger.debug(f"Raw response text: {clean_text}")
        
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
    Calls the agent to get insights for a single restaurant.
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
        logger.info("Initializing InMemoryRunner...")
        runner = InMemoryRunner(agent=root_agent)
        logger.info(f"InMemoryRunner initialized: {type(runner)}")
        
        # Use fhrsid as session_id to maintain distinct sessions per restaurant if needed, 
        # though we recreate runner each time here.
        session_id = f"session_{restaurant.get('fhrsid', 'unknown')}"
        logger.info(f"Using session_id: {session_id}")
        
        logger.info("Calling runner.run()...")
        events = runner.run(user_id="system", session_id=session_id, new_message=prompt)
        logger.info(f"runner.run() returned. Events type: {type(events)}")
        
        raw_text = ""
        for i, event in enumerate(events):
            logger.debug(f"Processing event {i}: {type(event)}")
            # Check if event has content and parts (based on ADK structure)
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        raw_text += part.text
        
        if not raw_text:
             logger.warning(f"Agent returned no text for {business_name}")
             return None

        logger.info(f"Agent returned text (length {len(raw_text)})")
        parsed = parse_agent_response(raw_text)
        parsed["raw_insight"] = raw_text
        parsed["fhrsid"] = restaurant.get("fhrsid")
        
        return parsed
        
    except Exception as e:
        logger.error(f"Error calling agent for {business_name}: {e}")
        logger.error(traceback.format_exc())
        return None