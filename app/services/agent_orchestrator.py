import json
import re
import logging
from typing import Dict, Any, Optional
import traceback
import requests
import google.auth
import google.auth.transport.requests

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

def get_auth_token():
    """Retrieves Google auth token."""
    credentials, project = google.auth.default()
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token

def get_agent_insight(restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calls the remote Vertex AI Agent to get insights for a single restaurant using direct REST API.
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
        token = get_auth_token()
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{AGENT_RESOURCE_ID}:streamQuery"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Payload for ADK Agent
        payload = {
            "input": {
                "message": {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            }
        }
        
        logger.info(f"Sending request to {url}")
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"Agent API Error: {response.status_code} - {response.text}")
            return None
            
        # Accumulate streaming response
        raw_text = ""
        for line in response.iter_lines():
            if line:
                # SSE format: "data: {...}"
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:] # Strip "data: "
                    try:
                        chunk_data = json.loads(json_str)
                        # Structure is likely specific to ADK agents or Reasoning Engine
                        # Need to traverse to find the text content.
                        # Usually it is in a structure related to the output.
                        # Let's dump whatever we get if we can't find it easily to debug,
                        # but for now let's try to find 'output' or similar.
                        
                        # Note: The streamQuery response is StreamQueryReasoningEngineResponse
                        # It might not have 'output' directly if it's SSE? 
                        # Or it wraps the response.
                        
                        # In many SSE implementations for Vertex, the chunk IS the response object.
                        # We are looking for something that resembles text.
                        # ADK output usually: ...
                        
                        # For now, let's just log chunks to debug if we don't get text immediately,
                        # but attempt to heuristic extract.
                        
                        # Assuming reasoning engine returns a value.
                        # If it's a string output:
                        # response might look like {"output": "some text"} or similar?
                        pass 
                        
                    except json.JSONDecodeError:
                        pass
        
        # Since parsing SSE chunks and reconstructing the full response object manually is complex 
        # and structure varies, for a quick fix, let's try non-streaming first? 
        # But deploy_agent uses streamQuery.
        
        # Actually, requests stream=True allows us to iterate lines. 
        # But if we just want the whole text, and if the response is not massive, we can just read it all?
        # But it's SSE, so `response.text` will be multiple JSON objects prefixed with `data:`.
        
        # Let's robustly parse SSE.
        full_response_text = ""
        # Re-iterating because I didn't consume it in the loop above (pass)
        
        # Wait, I cannot re-iterate generator. 
        # Let's do it properly.
        
        logger.info("Processing stream...")
        # requests.iter_lines() handles decoding.
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                        # Look for 'output'
                        # Structure might be: {"output": {"string_value": "..."}} or similar
                        
                        # Log one chunk for debugging (optional)
                        # logger.info(f"Chunk: {chunk}")
                        
                        # Heuristic extraction
                        if "output" in chunk:
                            output = chunk["output"]
                            if isinstance(output, str):
                                full_response_text += output
                            elif isinstance(output, dict):
                                # Check for protobuf Value fields
                                if "stringValue" in output:
                                    full_response_text += output["stringValue"]
                                # Add other cases if needed
                        
                        # Also check candidates/content if it's following GenerateContent style?
                        # Reasoning Engine usually returns 'output'.
                        
                    except Exception as e:
                        logger.warning(f"Error parsing chunk: {e}")

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