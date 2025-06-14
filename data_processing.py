import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Callable, Tuple, Optional
from bq_utils import ORIGINAL_COLUMNS_TO_KEEP

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
    Processes API data and identifies new establishments not present in the master data,
    ensuring no duplicates from the current API batch are added.

    Args:
        master_data: The current list of master restaurant data (used to check for existing FHRSIDs).
        api_data: The raw JSON data (as a dict) from the API.

    Returns:
        A list of newly added restaurant dictionaries, unique within this processing batch.
    """
    api_establishments = api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
    
    if api_establishments is None: 
        api_establishments = []
        st.warning("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not api_establishments: 
         st.info("API response contained no establishments in 'EstablishmentDetail'.")

    existing_fhrsid_set = set()
    for est in master_data:
        if isinstance(est, dict) and 'fhrsid' in est and est['fhrsid'] is not None:
            try:
                canonical_fhrsid = str(int(est['fhrsid']))
            except ValueError:
                canonical_fhrsid = str(est['fhrsid'])
                st.warning(f"FHRSID '{est['fhrsid']}' from master_data could not be converted to int. Using original string value for comparison.")
            existing_fhrsid_set.add(canonical_fhrsid)

    today_date = datetime.now().strftime("%Y-%m-%d")
    newly_added_restaurants: List[Dict[str, Any]] = []
    fhrsids_processed_in_this_batch = set() # New set to track FHRSIDs within the current batch

    for api_establishment in api_establishments:
        if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment and api_establishment['FHRSID'] is not None:
            original_api_fhrsid = api_establishment['FHRSID']
            try:
                canonical_api_fhrsid = str(int(original_api_fhrsid))
            except ValueError:
                canonical_api_fhrsid = str(original_api_fhrsid)
                st.warning(f"FHRSID '{original_api_fhrsid}' from API data could not be converted to int. Using original string value.")

            # Replace the original FHRSID with the canonical version
            api_establishment['FHRSID'] = canonical_api_fhrsid

            if canonical_api_fhrsid not in existing_fhrsid_set:
                # Check if this canonical FHRSID has already been processed in the current batch
                if canonical_api_fhrsid not in fhrsids_processed_in_this_batch:
                    api_establishment['first_seen'] = today_date
                    api_establishment['manual_review'] = "not reviewed"

                    # Filter and prepare the establishment data using ORIGINAL_COLUMNS_TO_KEEP
                    processed_establishment = {}
                    for key in ORIGINAL_COLUMNS_TO_KEEP:
                        if key in api_establishment:
                            processed_establishment[key] = api_establishment[key]
                        else:
                            # Ensure missing keys are explicitly set to None in the processed_establishment
                            processed_establishment[key] = None

                    # Ensure the 'FHRSID' in processed_establishment is the canonical_api_fhrsid.
                    # This is guaranteed because api_establishment['FHRSID'] was updated,
                    # and if 'FHRSID' is in ORIGINAL_COLUMNS_TO_KEEP, it will take the updated value.
                    # If 'FHRSID' were NOT in ORIGINAL_COLUMNS_TO_KEEP, we'd need:
                    # processed_establishment['FHRSID'] = canonical_api_fhrsid

                    newly_added_restaurants.append(processed_establishment)
                    fhrsids_processed_in_this_batch.add(canonical_api_fhrsid) # Add to batch tracking set
                else:
                    # Optional: Log that a duplicate FHRSID within the current API batch was skipped.
                    # Using print for now, can be changed to st.info or a more formal logger.
                    print(f"Skipping duplicate FHRSID {canonical_api_fhrsid} found within the current API batch (already processed).")
            # else:
                # Optional: Log that FHRSID was found in existing_fhrsid_set (already in BQ).
                # print(f"FHRSID {canonical_api_fhrsid} already exists in BigQuery master data. Skipping.")
    
    count_new_restaurants = len(newly_added_restaurants)
    if count_new_restaurants > 0:
        st.success(f"Processed API response. Identified {count_new_restaurants} unique new restaurant records to be added.")
    else:
        st.info("Processed API response. No new restaurant records identified (or all were duplicates within the batch or already in BigQuery).")

    return newly_added_restaurants

def load_data_from_csv(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Loads data from an uploaded CSV file into a Pandas DataFrame.
    Checks for 'fhrsid' column (case-insensitive) and converts it to string.

    Args:
        uploaded_file: A Streamlit UploadedFile object (or any file-like object
                       compatible with pandas.read_csv).

    Returns:
        A Pandas DataFrame if successful and 'fhrsid' column is present,
        otherwise None.
    """
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None

        # Search for 'fhrsid' column case-insensitively
        fhrsid_col_name = None
        for col in df.columns:
            if col.lower() == 'fhrsid':
                fhrsid_col_name = col
                break

        if fhrsid_col_name is None:
            st.error("The required 'fhrsid' column is missing in the uploaded CSV file.")
            return None

        # Convert fhrsid column to string
        df[fhrsid_col_name] = df[fhrsid_col_name].astype(str)

        # Rename column to 'fhrsid' if it's not already named that (for consistency)
        if fhrsid_col_name != 'fhrsid':
            df.rename(columns={fhrsid_col_name: 'fhrsid'}, inplace=True)

        return df

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or contains no data.")
        return None
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it's a valid CSV format.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the CSV file: {e}")
        return None
