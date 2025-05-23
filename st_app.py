import streamlit as st
import requests
import json
import pandas as pd
from google.cloud import storage
from datetime import datetime
import os
from google.cloud import bigquery
from typing import Tuple, Optional, List, Dict, Callable, Any


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
        return None
    
    parts = uri[5:].split("/", 1) # Remove "gs://" and split by the first "/"
    
    if len(parts) < 2 or not parts[0] or not parts[1]:
        # Must have a bucket and a blob name
        return None
        
    bucket_name = parts[0]
    blob_name = parts[1]
    
    return bucket_name, blob_name


def load_json_from_uri(uri: str):
    """
    Loads a JSON file from a given URI (GCS path or local file path).

    Args:
        uri: The URI of the JSON file (e.g., "gs://bucket/file.json" or "/path/to/file.json").

    Returns:
        A dictionary loaded from the JSON file, or None if an error occurs.
    """
    if uri.startswith("gs://"):
        parsed_uri = _parse_gcs_uri(uri)
        if parsed_uri is None:
            st.error(f"Invalid GCS URI format: {uri}")
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
    else:
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


def upload_to_gcs(gcs_uri: str, data_string: str, content_type: str = 'application/json') -> bool:
    """
    Uploads a string of data to a specified GCS URI.

    Args:
        gcs_uri: The GCS URI where the data should be uploaded (e.g., "gs://bucket-name/path/to/file.json").
        data_string: The string data to upload.
        content_type: The content type of the data (default is 'application/json').

    Returns:
        True if the upload was successful, False otherwise.
    """
    parsed_uri = _parse_gcs_uri(gcs_uri)
    if parsed_uri is None:
        st.error(f"Invalid GCS URI format: {gcs_uri}. It must start with gs:// and include a bucket and blob name.")
        return False
    
    bucket_name, blob_name = parsed_uri

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_string(data_string, content_type=content_type)
        # st.success(f"Successfully uploaded data to {gcs_uri}") # Optional: for direct use feedback
        return True
    except Exception as e: # Catches google.cloud.exceptions.GoogleCloudError and other potential errors
        st.error(f"Error uploading data to GCS ({gcs_uri}): {e}")
        return False


def fetch_api_data(longitude: float, latitude: float) -> Optional[Dict[str, Any]]:
    """
    Fetches data from the Food Standards Agency API.

    Args:
        longitude: The longitude for the API search.
        latitude: The latitude for the API search.

    Returns:
        A dictionary containing the JSON response from the API, or None if an error occurs.
    """
    api_url = f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/{longitude}/{latitude}/1/500/json"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: Could not fetch data from the API. Status Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: An exception occurred while making the API request: {e}")
        return None


def load_master_data(uri: str, load_json_func: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """
    Loads master restaurant data from a given URI using a provided loading function.

    Args:
        uri: The URI of the JSON file (GCS or local).
        load_json_func: The function to use for loading JSON from the URI (e.g., load_json_from_uri).

    Returns:
        A list of dictionaries representing the master restaurant data. Returns an empty list on failure.
    """
    if not uri:
        st.info("No master restaurant data URI provided. Starting with empty master restaurant data.")
        return []

    loaded_data = load_json_func(uri)

    if loaded_data is None:
        # load_json_func is expected to call st.error for specific loading issues.
        st.warning(f"Failed to load master restaurant data from {uri} (or it was empty/invalid). Proceeding with empty master restaurant data.")
        return []
    
    if isinstance(loaded_data, list):
        if loaded_data:
            st.success(f"Successfully loaded master restaurant data with {len(loaded_data)} records from {uri}.")
        else:
            st.warning(f"Master restaurant data loaded from {uri}, but it's empty.")
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
    
    if api_establishments is None: # Explicitly check for None, as an empty list is valid
        api_establishments = []
        st.warning("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not api_establishments: # Check if the list is empty
         st.info("API response contained no establishments in 'EstablishmentDetail'.")

    existing_fhrsid_set = {est['FHRSID'] for est in master_data if isinstance(est, dict) and 'FHRSID' in est}
    today_date = datetime.now().strftime("%Y-%m-%d")
    new_restaurants_added_count = 0

    for api_establishment in api_establishments:
        if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
            if api_establishment['FHRSID'] not in existing_fhrsid_set:
                api_establishment['first_seen'] = today_date
                master_data.append(api_establishment)
                existing_fhrsid_set.add(api_establishment['FHRSID'])
                new_restaurants_added_count += 1
    
    st.success(f"Processed API response. Added {new_restaurants_added_count} new restaurant records. Total unique records: {len(master_data)}")
    return master_data, new_restaurants_added_count


def display_data(data_to_display: List[Dict[str, Any]]):
    """
    Displays the given data using Streamlit, primarily as a Pandas DataFrame.

    Args:
        data_to_display: A list of dictionaries to display.
    """
    try:
        if not data_to_display:
            st.warning("No restaurant data to display (master restaurant data is empty after processing).")
            return

        # Ensure all items are dictionaries before normalization
        valid_items_for_df = [item for item in data_to_display if isinstance(item, dict)]
        
        if not valid_items_for_df:
            st.warning("Master restaurant data contains no dictionary items, cannot display as table.")
        elif len(valid_items_for_df) < len(data_to_display):
            st.warning(f"Some items in the master restaurant data were not dictionaries and were excluded from the table display. Displaying {len(valid_items_for_df)} records.")
        
        if valid_items_for_df: # Proceed only if there's something to show
            df = pd.json_normalize(valid_items_for_df)
            st.dataframe(df)
        
    except Exception as e: 
        st.error(f"Error displaying DataFrame from master restaurant data: {e}")
        st.info("Attempting to show raw master restaurant data as JSON.")
        try:
            st.json(data_to_display)
        except Exception as json_e:
            st.error(f"Could not even display master restaurant data as JSON: {json_e}")


def write_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str):
    """
    Writes a Pandas DataFrame to a BigQuery table.

    Args:
        df: The DataFrame to write.
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.

    Returns:
        True if the write operation was successful, False otherwise.
    """
    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        column_name_character_map="V2",
    )
    
    try:
        job = client.load_table_from_dataframe(df, table_ref_str, job_config=job_config)
        job.result()  # Wait for the job to complete
        st.success(f"Successfully wrote data to BigQuery table {table_ref_str}. Overwritten if table existed.")
        return True
    except Exception as e:
        st.error(f"Error writing data to BigQuery table {table_ref_str}: {e}")
        return False


# Set the title of the Streamlit app
st.title("Food Standards Agency API Explorer")

# Create input fields for longitude and latitude
longitude = st.number_input("Enter Longitude", format="%.3f")
latitude = st.number_input("Enter Latitude", format="%.3f")

# Create an input field for the GCS destination folder URI
gcs_destination_uri = st.text_input("Enter GCS destination folder for the scan (e.g., gs://bucket-name/scans-folder/)")

# Create an input field for the master restaurant list URI
master_list_uri = st.text_input("Enter Master Restaurant Data URI (JSON) (e.g., gs://bucket/file.json or /path/to/file.json)")

# Create an input field for the GCS destination for the master restaurant data
gcs_master_dictionary_output_uri = st.text_input("Enter GCS URI for Master Restaurant Data Output (e.g., gs://bucket-name/path/filename.json)")

# Create an input field for the BigQuery full path
bq_full_path = st.text_input("Enter BigQuery Table Path (project.dataset.table)")

# Create a button to trigger the API call
if st.button("Fetch Data"):
    # 1. Fetch API Data
    api_data = fetch_api_data(longitude, latitude)

    if api_data:
        # 2. Load Master Restaurant Data
        # Pass the actual load_json_from_uri function as the callable
        master_restaurant_data = load_master_data(master_list_uri, load_json_from_uri)

        # 3. Process API Response and Update Master Restaurant Data
        master_restaurant_data, _ = process_and_update_master_data(master_restaurant_data, api_data)
        
        # 4. Upload Raw API Response to gcs_destination_uri (folder)
        if gcs_destination_uri:
            current_date = datetime.now().strftime("%Y-%m-%d")
            api_response_filename = f"api_response_{current_date}.json"
            
            gcs_destination_uri_folder = gcs_destination_uri
            if not gcs_destination_uri_folder.endswith('/'):
                gcs_destination_uri_folder += '/'
            
            full_gcs_path_api_response = f"{gcs_destination_uri_folder}{api_response_filename}"

            if upload_to_gcs(full_gcs_path_api_response, json.dumps(api_data, indent=4)):
                st.success(f"Successfully uploaded raw API response to {full_gcs_path_api_response}")
            # upload_to_gcs handles its own st.error messages on failure
        
        # 5. Upload Master Restaurant Data to gcs_master_dictionary_output_uri (full file path)
        if gcs_master_dictionary_output_uri:
            if upload_to_gcs(gcs_master_dictionary_output_uri, json.dumps(master_restaurant_data, indent=4)):
                st.success(f"Successfully uploaded master restaurant data to {gcs_master_dictionary_output_uri}")
            # upload_to_gcs handles its own st.error messages on failure

        # 6. Display Data
        display_data(master_restaurant_data)

        # 7. Write to BigQuery
        if api_data: # Only proceed if API data was fetched successfully
            if master_restaurant_data:
                df_to_load = pd.json_normalize([item for item in master_restaurant_data if isinstance(item, dict)])
                if df_to_load is not None and not df_to_load.empty:
                    if bq_full_path:
                        try:
                            path_parts = bq_full_path.split('.')
                            if len(path_parts) == 3: # Check if split produced three parts
                                project_id, dataset_id, table_id = path_parts
                                if len(project_id) > 0 and len(dataset_id) > 0 and len(table_id) > 0: # Basic check for non-empty parts
                                    st.info(f"Attempting to write to BigQuery table: {bq_full_path}")
                                    write_to_bigquery(df_to_load, project_id, dataset_id, table_id)
                                else: # Handles cases like ".dataset.table" or "project.."
                                    st.error(f"Invalid BigQuery Table Path format. Each part of 'project.dataset.table' must be non-empty. Got: '{bq_full_path}'. Skipping BigQuery write.")
                            else: # Handles cases where split does not return 3 parts (e.g. "proj.dataset" or "proj.dataset.table.extra")
                                st.error(f"Invalid BigQuery Table Path format. Expected 'project.dataset.table' (3 parts), but got {len(path_parts)} part(s) from '{bq_full_path}'. Skipping BigQuery write.")
                        except ValueError: # Should not be strictly necessary due to the len check, but good for safety.
                            st.error(f"Invalid BigQuery Table Path format. Expected 'project.dataset.table'. Error during parsing '{bq_full_path}'. Skipping BigQuery write.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during BigQuery operations setup or write: {e}")
                    else:
                        st.warning("Master data is ready, but the BigQuery Table Path is missing. Skipping BigQuery write.")
                else:
                    st.warning("Master data is empty or contains no valid records after processing. Skipping BigQuery write.")
            else:
                st.warning("Master data is empty (was not loaded or was cleared). Skipping BigQuery write.")
        # else: # api_data is False. fetch_api_data already displayed an error.
        # else: # This would mean api_data is False. fetch_api_data handles its own error message.
            # No explicit message needed here if api_data is None, as fetch_api_data handles it.
            # However, if only bq_table_name was provided without api_data, the old message was:
            # "No API data fetched, skipping BigQuery write even though a table name was provided."
            # This is now implicitly covered. If api_data is False, the first 'if' fails.
            # If api_data is True, but BQ details are missing, the 'elif api_data:' handles it.

    # If api_data is None, fetch_api_data already displayed an error message.
