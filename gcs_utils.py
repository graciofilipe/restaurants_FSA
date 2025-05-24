import json
import streamlit as st
from google.cloud import storage
from typing import Tuple, Optional, Dict, Any

def _parse_gcs_uri(uri: str) -> Optional[Tuple[str, str]]:
    """
    Parses a GCS URI into bucket name and blob name.

    Args:
        uri: The GCS URI string (e.g., "gs://bucket/file.json").

    Returns:
        A tuple (bucket_name, blob_name) if parsing is successful, 
        or None if the URI is invalid.
    """
    if not uri.startswith("gs://"):
        st.error(f"Invalid GCS URI: {uri}. Must start with 'gs://'.") # Added error message as per initial plan
        return None
    
    parts = uri[5:].split("/", 1) # Remove "gs://" and split by the first "/"
    
    if len(parts) < 2 or not parts[0] or not parts[1]:
        # Must have a bucket and a blob name
        st.error(f"Invalid GCS URI: {uri}. Must contain bucket and blob name.") # Added error message
        return None
        
    bucket_name = parts[0]
    blob_name = parts[1]
    
    return bucket_name, blob_name

def upload_to_gcs(data: Dict[str, Any], destination_uri: str) -> Optional[str]:
    """Uploads a dictionary as a JSON file to Google Cloud Storage.

    Args:
        data: The dictionary to upload.
        destination_uri: The GCS URI to upload to (e.g., gs://bucket/blob.json).

    Returns:
        The GCS URI of the uploaded file, or None if upload fails.
    """
    parsed_uri = _parse_gcs_uri(destination_uri)
    if parsed_uri is None:
        # _parse_gcs_uri already calls st.error
        return None
    
    bucket_name, blob_name = parsed_uri

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        data_string = json.dumps(data, indent=4) # Convert dict to JSON string
        
        blob.upload_from_string(data_string, content_type='application/json')
        # st.success(f"Successfully uploaded data to {destination_uri}") # Optional: for direct use feedback
        return destination_uri # Return the URI on success
    except Exception as e: # Catches google.cloud.exceptions.GoogleCloudError and other potential errors
        st.error(f"Error uploading data to GCS ({destination_uri}): {e}")
        return None

def load_json_from_gcs(uri: str) -> Optional[Dict[str, Any]]:
    """Loads a JSON file from GCS.

    Args:
        uri: The GCS URI of the JSON file.

    Returns:
        The loaded JSON data as a dictionary, or None if loading fails.
    """
    parsed_uri = _parse_gcs_uri(uri)
    if parsed_uri is None:
        # _parse_gcs_uri handles st.error
        return None
    
    bucket_name, blob_name = parsed_uri
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            st.error(f"Error: GCS file not found at {uri}")
            return None

        data_string = blob.download_as_string()
        return json.loads(data_string)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from GCS file {uri}: {e}")
        return None
    except Exception as e: # Catching a broader range of GCS client/permission errors
        st.error(f"Error accessing GCS file {uri}: {e}")
        return None
