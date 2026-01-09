import json
import re
import logging
from typing import Dict, Any, Optional
import traceback
import vertexai
from vertexai import Client

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
            start = clean_text.find("{")
            end = clean_text.rfind("}")
            if start != -1 and end != -1:
                 clean_text = clean_text[start:end+1]
        
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
    Calls the remote Vertex AI Agent to get insights for a single restaurant using the cloud-native Client SDK.
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
        logger.info(f"Initializing Vertex AI Client (Project: {PROJECT_ID}, Location: {LOCATION})...")
        # Ensure we are initialized for overall vertexai
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Use the newer Client interface
        client = Client(project=PROJECT_ID, location=LOCATION)
        
        logger.info(f"Fetching Agent Engine: {AGENT_RESOURCE_ID}...")
        agent = client.agent_engines.get(name=AGENT_RESOURCE_ID)
        
        logger.info("Calling stream_query...")
        # For ADK agents, we use the message parameter. 
        # A user_id is recommended for ADK agents.
        response_stream = agent.stream_query(
            message=prompt,
            user_id="fsa_reviewer_app"
        )
        
        full_response_text = ""
        
        # Process the stream
        for chunk in response_stream:
            # Chunk is typically a dictionary containing 'content'
            if isinstance(chunk, dict) and "content" in chunk:
                content = chunk["content"]
                if "parts" in content:
                    for part in content["parts"]:
                        if "text" in part:
                            full_response_text += part["text"]
            elif hasattr(chunk, 'text'): # Fallback for object-like chunk
                full_response_text += chunk.text
            elif isinstance(chunk, str): # Fallback for plain string chunk
                full_response_text += chunk

        logger.info(f"Accumulated response length: {len(full_response_text)}")
        logger.info(f"Raw response (first 100 chars): {full_response_text[:100]}")
        
        if not full_response_text:
             logger.warning(f"Agent returned no text for {business_name}")
             return None

        parsed = parse_agent_response(full_response_text)
        parsed["raw_insight"] = full_response_text
        parsed["fhrsid"] = restaurant.get("fhrsid")
        
        return parsed
        
    except Exception as e:
        logger.error(f"Error calling agent for {business_name}: {e}")
        logger.error(traceback.format_exc())
        return None
