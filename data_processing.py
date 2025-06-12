import json
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Callable, Tuple, Optional

def load_json_from_local_file_path(uri: str) -> Optional[Dict[str, Any]]:
    """
    Loads a JSON file from a local file path.

    Args:
        uri: The local file path of the JSON file (e.g., "/path/to/file.json").

    Returns:
        A dictionary loaded from the JSON file, or None if an error occurs.
    """
    try:
        with open(uri, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Local file not found at {uri}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from local file {uri}: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading local file {uri}: {e}")
        return None

def load_master_data(uri: str, load_json_func: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """
    Loads master restaurant data from a given URI using a provided loading function.

    Args:
        uri: The URI of the JSON file (GCS or local).
        load_json_func: The function to use for loading JSON from the URI.

    Returns:
        A list of dictionaries representing the master restaurant data. Returns an empty list on failure.
    """
    if not uri:
        st.info("No master restaurant data URI provided. Starting with empty master restaurant data.")
        return []

    loaded_data = load_json_func(uri)

    if loaded_data is None:
        st.warning(f"Failed to load master restaurant data from {uri} (or it was empty/invalid). Proceeding with empty master restaurant data.")
        return []
    
    if isinstance(loaded_data, list):
        if loaded_data:
            st.success(f"Successfully loaded master restaurant data with {len(loaded_data)} records from {uri}.")
        else:
            st.warning(f"Master restaurant data loaded from {uri}, but it's empty.")
    if isinstance(loaded_data, list):
        for restaurant in loaded_data:
            if isinstance(restaurant, dict) and restaurant.get("manual_review") is None:
                restaurant["manual_review"] = "not reviewed"
            if isinstance(restaurant, dict) and restaurant.get("gemini_insights") is None:
                restaurant["gemini_insights"] = None
        return loaded_data
    else:
        st.warning(f"Data loaded from {uri} is not in the expected list format. Type found: {type(loaded_data)}. Proceeding with empty master restaurant data.")
        return []

def process_and_update_master_data(master_data: List[Dict[str, Any]], api_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Processes API data and updates the master list of restaurants.

    Args:
        master_data: The current list of master restaurant data.
        api_data: The raw JSON data (as a dict) from the API.

    Returns:
        A tuple containing the updated master_data list and the count of new restaurants added.
    """
    api_establishments = api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
    
    if api_establishments is None: 
        api_establishments = []
        st.warning("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not api_establishments: 
         st.info("API response contained no establishments in 'EstablishmentDetail'.")

    existing_fhrsid_set = {est['FHRSID'] for est in master_data if isinstance(est, dict) and 'FHRSID' in est}
    today_date = datetime.now().strftime("%Y-%m-%d")
    new_restaurants_added_count = 0

    for api_establishment in api_establishments:
        if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
            if api_establishment['FHRSID'] not in existing_fhrsid_set:
                api_establishment['first_seen'] = today_date
                api_establishment['manual_review'] = "not reviewed"
                api_establishment['gemini_insights'] = None
                master_data.append(api_establishment)
                existing_fhrsid_set.add(api_establishment['FHRSID'])
                new_restaurants_added_count += 1
    
    st.success(f"Processed API response. Added {new_restaurants_added_count} new restaurant records. Total unique records: {len(master_data)}")
    return master_data, new_restaurants_added_count
