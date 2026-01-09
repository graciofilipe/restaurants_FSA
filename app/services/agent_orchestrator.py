import json
import re
import logging
from typing import Dict, Any, Optional
import traceback
import vertexai
from vertexai.preview import reasoning_engines
from google.cloud.aiplatform_v1beta1.types import QueryReasoningEngineRequest

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
        
        logger.info("Inspecting operation schemas...")
        try:
            schemas = remote_agent.operation_schemas()
            logger.info(f"Operation Schemas: {schemas}")
        except Exception as e:
            logger.warning(f"Could not fetch operation schemas: {e}")

        response = None
        if hasattr(remote_agent, 'query'):
            logger.info("Using standard .query() method.")
            response = remote_agent.query(input=prompt)
        else:
            logger.warning("Method .query() missing (likely due to async mode mismatch). Attempting manual client call.")
            
            # Assuming the method name is 'query' based on standard pattern
            # Construct the input payload. 
            request_input = {"input": prompt} 
            
            request = QueryReasoningEngineRequest(
                name=remote_agent.resource_name,
                input=request_input,
                class_method="query"
            )
            
            api_response = remote_agent.execution_api_client.query_reasoning_engine(request=request)
            
            # The API response output is a google.protobuf.Struct or similar wrapped value.
            # We need to extract it.
            # Usually api_response.output contains the result.
            if hasattr(api_response, 'output'):
                response = api_response.output
            else:
                logger.error(f"API response has no 'output' field: {api_response}")
                return None

        logger.info(f"Agent query returned. Type: {type(response)}")
        
        raw_text = str(response)
        
        # Handle dict-like response (common if it returns JSON structure)
        if isinstance(response, dict):
             raw_text = json.dumps(response)
        # Handle if it's a Value/Struct from protobuf
        elif hasattr(response, 'items'): 
             # Rough check for dict-like behavior from protobuf map
             try:
                 import google.protobuf.json_format
                 # If it's a message, convert. If it's a MapComposite, cast to dict.
                 raw_text = json.dumps(dict(response))
             except:
                 pass

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