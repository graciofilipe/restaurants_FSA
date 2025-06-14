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

def load_master_data(project_id: str, dataset_id: str, table_id: str, load_bq_func: Callable[[str, str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Loads master restaurant data from a BigQuery table.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        load_bq_func: The function to use for loading data from BigQuery.
                      Expected to be bq_utils.load_all_data_from_bq.

    Returns:
        A list of dictionaries representing the master restaurant data. Returns an empty list on failure.
    """
    st.info(f"Loading master restaurant data from BigQuery table: {project_id}.{dataset_id}.{table_id}")

    try:
        loaded_data = load_bq_func(project_id, dataset_id, table_id)
    except Exception as e:
        st.error(f"An error occurred while calling load_bq_func for {project_id}.{dataset_id}.{table_id}: {e}")
        loaded_data = [] # Ensure loaded_data is an empty list on exception

    if loaded_data is None: # load_all_data_from_bq is designed to return [] on error, but good to be defensive
        st.warning(f"Failed to load master restaurant data from BigQuery table {project_id}.{dataset_id}.{table_id} (function returned None). Proceeding with empty master restaurant data.")
        return []
    
    if isinstance(loaded_data, list):
        if loaded_data:
            st.success(f"Successfully loaded {len(loaded_data)} records from BigQuery table {project_id}.{dataset_id}.{table_id}.")
        else:
            st.info(f"Master restaurant data loaded from BigQuery table {project_id}.{dataset_id}.{table_id}, but the table is empty or returned no data.")

        # Retain existing logic for default 'manual_review'
        for restaurant in loaded_data:
            if isinstance(restaurant, dict) and restaurant.get("manual_review") is None:
                restaurant["manual_review"] = "not reviewed"
        return loaded_data
    else:
        # This case should ideally not be reached if load_bq_func adheres to its return type List[Dict[str, Any]]
        st.error(f"Data loaded from BigQuery table {project_id}.{dataset_id}.{table_id} is not in the expected list format. Type found: {type(loaded_data)}. Proceeding with empty master restaurant data.")
        return []

def process_and_update_master_data(master_data: List[Dict[str, Any]], api_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes API data and identifies new establishments not present in the master data.

    Args:
        master_data: The current list of master restaurant data (used to check for existing FHRSIDs).
        api_data: The raw JSON data (as a dict) from the API.

    Returns:
        A list of newly added restaurant dictionaries.
    """
    api_establishments = api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
    
    if api_establishments is None: 
        api_establishments = []
        st.warning("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not api_establishments: 
         st.info("API response contained no establishments in 'EstablishmentDetail'.")

    existing_fhrsid_set = {est['FHRSID'] for est in master_data if isinstance(est, dict) and 'FHRSID' in est}
    today_date = datetime.now().strftime("%Y-%m-%d")
    newly_added_restaurants: List[Dict[str, Any]] = []

    for api_establishment in api_establishments:
        if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
            # Ensure FHRSID is an integer for comparison and storage
            fhrsid_int = api_establishment['FHRSID'] # Assuming FHRSID from API is already an int or convertible
            if not isinstance(fhrsid_int, int):
                try:
                    fhrsid_int = int(fhrsid_int)
                except (ValueError, TypeError):
                    st.warning(f"Could not convert FHRSID '{fhrsid_int}' to int for establishment: {api_establishment.get('BusinessName', 'N/A')}. Skipping this record.")
                    continue # Skip if FHRSID cannot be an integer

            if fhrsid_int not in existing_fhrsid_set:
                api_establishment['FHRSID'] = fhrsid_int # Ensure the integer version is stored
                api_establishment['first_seen'] = today_date
                api_establishment['manual_review'] = "not reviewed"
                newly_added_restaurants.append(api_establishment)
                # Note: We do not add to existing_fhrsid_set here as master_data is not modified by this function.
                # If multiple identical new FHRSIDs are in api_data, they will all be added. This is consistent with previous behavior of adding all to master.
    
    count_new_restaurants = len(newly_added_restaurants)
    if count_new_restaurants > 0:
        st.success(f"Processed API response. Identified {count_new_restaurants} new restaurant records to be added.")
    else:
        st.info("Processed API response. No new restaurant records identified.")

    return newly_added_restaurants
