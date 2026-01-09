import json
import re
import logging
from typing import Dict, Any, Optional
import traceback
import vertexai
from vertexai.preview import reasoning_engines
from google.cloud.aiplatform_v1beta1.types import QueryReasoningEngineRequest, StreamQueryReasoningEngineRequest

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

        raw_text = ""
        
        if hasattr(remote_agent, 'query'):
            logger.info("Using standard .query() method.")
            response = remote_agent.query(input=prompt)
            raw_text = str(response)
            if hasattr(response, 'text'):
                 raw_text = response.text
            elif isinstance(response, dict):
                 raw_text = json.dumps(response)
        else:
            logger.warning("Method .query() missing. Attempting manual client call with stream_query.")
            
            # Construct the input conforming to what ADK Agent expects (message object)
            method_kwargs = {
                "message": {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            }
            
            request = StreamQueryReasoningEngineRequest(
                name=remote_agent.resource_name,
                input=method_kwargs,
                class_method="stream_query"
            )
            
            logger.info(f"Calling stream_query_reasoning_engine with class_method='stream_query'")
            response_stream = remote_agent.execution_api_client.stream_query_reasoning_engine(request=request)
            
            # Accumulate text from stream
            for chunk in response_stream:
                if hasattr(chunk, 'output') and chunk.output:
                    # chunk.output is a google.protobuf.Value
                    # We usually expect a string or a dict. 
                    # If it's a string value, it might be directly accessible if mapped, 
                    # but typically we need to access the value.
                    # Based on observation, it might be nested.
                    # Let's try to extract text conservatively.
                    
                    # If it's a simple string yield
                    # In python client, yield_parsed_json does some work. 
                    # Here we are working with raw proto/gapic object.
                    
                    # Debug log the chunk structure once
                    # logger.info(f"Chunk received: {chunk}")
                    
                    # Assuming it returns parts of text or full objects.
                    # For now, let's cast to string whatever we get and append if it seems like content.
                    # ADK agents often stream partial text.
                    
                    # We can use the helper from reasoning_engines if available, but it is internal.
                    # Let's try to extract text from the protobuf Value.
                    
                    # A robust way to extract value from protobuf Value:
                    import google.protobuf.json_format
                    try:
                        # Convert to python object
                        val = None
                        # There isn't a direct to_py helper on the chunk itself, but chunk.output is a Value
                        # We can try to serialize/deserialize or inspect 'string_value' etc.
                        # But Value class has 'string_value', 'struct_value', etc.
                        # It is a google.protobuf.struct_pb2.Value
                        
                        kind = chunk.output.WhichOneof("kind")
                        if kind == "string_value":
                            val = chunk.output.string_value
                        elif kind == "struct_value":
                            # If it returns a struct, maybe convert to JSON string?
                            val = json.dumps(dict(chunk.output.struct_value.fields))
                        elif kind == "number_value":
                            val = str(chunk.output.number_value)
                        
                        if val:
                            raw_text += val
                    except Exception as parse_err:
                        logger.warning(f"Error parsing chunk: {parse_err}")

        logger.info(f"Agent response received. Length: {len(raw_text)}")
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