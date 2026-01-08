import json
import re
import logging
from typing import Dict, Any, Optional
from app.maps_agent.agent import root_agent

logger = logging.getLogger(__name__)

def parse_agent_response(response_text: str) -> Dict[str, Any]:
    """
    Parses the raw text response from the agent to extract structured data.
    Expected format: JSON or Markdown code block with JSON.
    Fields: cuisine_type (str), review_count (int), average_rating (float).
    """
    default_result = {
        "cuisine_type": None,
        "review_count": None,
        "average_rating": None
    }
    
    if not response_text:
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
            
            return {
                "cuisine_type": data.get("cuisine_type"),
                "review_count": data.get("review_count"),
                "average_rating": data.get("average_rating")
            }
        else:
            logger.warning(f"No JSON found in agent response: {response_text[:100]}...")
            return default_result
            
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON from agent response: {response_text[:100]}...")
        return default_result
    except Exception as e:
        logger.error(f"Error parsing agent response: {e}")
        return default_result

def get_agent_insight(restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calls the agent to get insights for a single restaurant.
    """
    business_name = restaurant.get("businessname")
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
        response = root_agent.chat(prompt)
        raw_text = response.text
        
        parsed = parse_agent_response(raw_text)
        parsed["raw_insight"] = raw_text
        parsed["fhrsid"] = restaurant.get("fhrsid")
        
        return parsed
        
    except Exception as e:
        logger.error(f"Error calling agent for {business_name}: {e}")
        return None
