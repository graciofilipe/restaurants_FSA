import streamlit as st
import requests
import json
import pandas as pd
from google.cloud import storage
from datetime import datetime
import os
from google.cloud import bigquery
from typing import Tuple, Optional, List, Dict, Callable, Any
import re


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


def fetch_api_data(longitude: float, latitude: float, max_results: int) -> Optional[Dict[str, Any]]:
    """
    Fetches data from the Food Standards Agency API.

    Args:
        longitude: The longitude for the API search.
        latitude: The latitude for the API search.
        max_results: The maximum number of results to fetch from the API.

    Returns:
        A dictionary containing the JSON response from the API, or None if an error occurs.
    """
    api_url = f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/{longitude}/{latitude}/1/{max_results}/json"
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


def sanitize_column_name(column_name: str) -> str:
    """
    Sanitizes a column name for BigQuery compatibility.
    - Removes spaces, periods, '@' signs, and dashes.
    - Converts to lowercase.
    - Replaces sequences of non-alphanumeric characters (excluding underscores) with a single underscore.
    - Ensures the name doesn't start or end with an underscore.
    - Handles potential leading characters like '?' or '@' from json_normalize.
    """
    # Replace problematic characters with underscore or remove them
    # Order matters: handle specific removals before general non-alphanumeric replacement
    name = column_name.replace(' ', '_')
    name = name.replace('.', '')  # Remove periods
    name = name.replace('@', '')  # Remove @
    name = name.replace('-', '_') # Replace dash with underscore
    
    name = name.lower()
    
    # Remove any leading characters that are not alphanumeric or underscore
    # This helps with characters like '?' often added by json_normalize
    if name and not name[0].isalnum() and name[0] != '_':
        name = name[1:]

    # Replace any remaining sequence of non-alphanumeric characters (except underscore) with a single underscore
    name = re.sub(r'[^a-z0-9_]+', '_', name)
    
    # Ensure it doesn't start or end with an underscore
    name = name.strip('_')
    
    # If the name becomes empty after stripping (e.g. was "___"), provide a default
    if not name:
        return "unnamed_column"
        
    return name


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
    # Sanitize column names
    original_columns = df.columns.tolist()
    sanitized_columns = [sanitize_column_name(col) for col in original_columns]
    
    # Check for duplicate sanitized column names and handle them if necessary
    # For now, we assume sanitize_column_name produces sufficiently unique names
    # or that BigQuery handles minor residual issues if any.
    # If critical, a more robust de-duplication strategy would be added here.
    df.columns = sanitized_columns
    
    # Logging the change for traceability (optional, but good practice)
    # st.info(f"Original columns: {original_columns}")
    # st.info(f"Sanitized columns for BigQuery: {sanitized_columns}")

    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        # Ensure V2 map is used as sanitize_column_name is designed for it.
        # BQ errors if it sees characters like '.' if not using V2.
        column_name_character_map="V2", 
    )
    
    try:
        job = client.load_table_from_dataframe(df, table_ref_str, job_config=job_config)
        job.result()  # Wait for the job to complete
        st.success(f"Successfully wrote data to BigQuery table {table_ref_str} with sanitized column names. Overwritten if table existed.")
        return True
    except Exception as e:
        st.error(f"Error writing data to BigQuery table {table_ref_str}: {e}")
        return False


# Set the title of the Streamlit app
st.title("Food Standards Agency API Explorer")

# Create input field for coordinate pairs
coordinate_pairs_input = st.text_area("Enter longitude,latitude pairs (one per line):")

# Create an input field for max results
max_results_input = st.number_input("Enter Max Results for API Call", min_value=1, max_value=5000, value=200)

# Create an input field for the GCS destination folder URI
gcs_destination_uri = st.text_input("Enter GCS destination folder for the scan (e.g., gs://bucket-name/scans-folder/)")

# Create an input field for the master restaurant list URI
master_list_uri = st.text_input("Enter Master Restaurant Data URI (JSON) (e.g., gs://bucket/file.json or /path/to/file.json)")

# Create an input field for the GCS destination for the master restaurant data
gcs_master_dictionary_output_uri = st.text_input("Enter GCS URI for Master Restaurant Data Output (e.g., gs://bucket-name/path/filename.json)")

# Create an input field for the BigQuery full path
bq_full_path = st.text_input("Enter BigQuery Table Path (project.dataset.table)")


def handle_fetch_data_action(
    coordinate_pairs_str: str, 
    max_results: int, 
    gcs_destination_uri_str: str, 
    master_list_uri_str: str, 
    gcs_master_output_uri_str: str, 
    bq_full_path_str: str
) -> List[Dict[str, Any]]:
    """
    Handles the core logic of fetching, processing, and saving data 
    when the 'Fetch Data' button is clicked.
    """
    coordinate_lines = coordinate_pairs_str.strip().split('\n')
    valid_coords = []
    for i, line in enumerate(coordinate_lines):
        line = line.strip()
        if not line:
            continue
        try:
            lon_str, lat_str = line.split(',')
            lon = float(lon_str.strip())
            lat = float(lat_str.strip())
            valid_coords.append((lon, lat))
        except ValueError:
            st.error(f"Error parsing line {i+1}: '{line}'. Expected 'longitude,latitude'. Skipping this line.")
            continue

    if not valid_coords:
        st.error("No valid coordinate pairs found. Please enter coordinates in 'longitude,latitude' format, one per line.")
        st.stop() # Stop further processing in this block

    all_api_establishments = []
    total_results_from_all_calls = 0

    st.info(f"Found {len(valid_coords)} valid coordinate pairs. Fetching data for each...")

    for lon, lat in valid_coords:
        st.write(f"Fetching data for Longitude: {lon}, Latitude: {lat}...")
        api_response = fetch_api_data(lon, lat, max_results_input)
        
        if api_response:
            establishments_list = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
            if establishments_list is None: # Ensure it's a list
                establishments_list = []
            
            num_results_this_call = len(establishments_list)
            total_results_from_all_calls += num_results_this_call
            st.info(f"API call for ({lon}, {lat}) returned {num_results_this_call} establishments.")

            if num_results_this_call == max_results_input:
                st.warning(f"Warning for ({lon}, {lat}): The API returned {num_results_this_call} results, matching `max_results`. Results might be capped.")
            
            all_api_establishments.extend(establishments_list)
        # fetch_api_data itself will show an error if the call fails

    if not all_api_establishments:
        st.info("No establishments found from any of the API calls. Nothing to process further.")
        st.stop()

    st.success(f"Total establishments fetched from all API calls: {len(all_api_establishments)}")
    
    # This combined_api_data will be used for subsequent processing steps
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}
    
    # For compatibility with existing logic that expects an 'api_data' variable for GCS upload and BQ.
    # This means the raw GCS upload will store the *combined* data.
    api_data = combined_api_data 

    # 2. Load Master Restaurant Data
    # Pass the actual load_json_from_uri function as the callable
    master_restaurant_data = load_master_data(master_list_uri, load_json_from_uri)

    # 3. Process API Response (Combined) and Update Master Restaurant Data
    master_restaurant_data, new_additions_count = process_and_update_master_data(master_restaurant_data, combined_api_data)
    # The success message from process_and_update_master_data already states new additions and total.
    
    # 4. Upload Combined Raw API Response to gcs_destination_uri_str (folder)
    # The variable 'api_data' now holds 'combined_api_data'
    if gcs_destination_uri_str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Consider a more descriptive filename if multiple coords are common, e.g., including hash or range
        api_response_filename = f"combined_api_response_{current_date}.json" 
        
        gcs_destination_uri_folder = gcs_destination_uri_str
        if not gcs_destination_uri_folder.endswith('/'):
            gcs_destination_uri_folder += '/'
        
        full_gcs_path_api_response = f"{gcs_destination_uri_folder}{api_response_filename}"

        if upload_to_gcs(full_gcs_path_api_response, json.dumps(api_data, indent=4)):
            st.success(f"Successfully uploaded combined raw API response to {full_gcs_path_api_response}")
        # upload_to_gcs handles its own st.error messages on failure
    
    # 5. Upload Master Restaurant Data to gcs_master_output_uri_str (full file path)
    if gcs_master_output_uri_str:
        if upload_to_gcs(gcs_master_output_uri_str, json.dumps(master_restaurant_data, indent=4)):
            st.success(f"Successfully uploaded master restaurant data to {gcs_master_output_uri_str}")
        # upload_to_gcs handles its own st.error messages on failure

    # 6. Display Data
    display_data(master_restaurant_data)

    # 7. Write to BigQuery
    # The variable 'api_data' (holding combined_api_data) can be used to check if any data was fetched.
    if api_data and api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail'): 
        if master_restaurant_data: # Check if master data (potentially enriched) exists
            df_to_load = pd.json_normalize([item for item in master_restaurant_data if isinstance(item, dict)])
            if df_to_load is not None and not df_to_load.empty:
                if bq_full_path_str:
                    try:
                        path_parts = bq_full_path_str.split('.')
                        if len(path_parts) == 3: 
                            project_id, dataset_id, table_id = path_parts
                            if len(project_id) > 0 and len(dataset_id) > 0 and len(table_id) > 0:
                                st.info(f"Attempting to write to BigQuery table: {bq_full_path_str}")
                                write_to_bigquery(df_to_load, project_id, dataset_id, table_id)
                            else: 
                                st.error(f"Invalid BigQuery Table Path format. Each part of 'project.dataset.table' must be non-empty. Got: '{bq_full_path_str}'. Skipping BigQuery write.")
                        else: 
                            st.error(f"Invalid BigQuery Table Path format. Expected 'project.dataset.table' (3 parts), but got {len(path_parts)} part(s) from '{bq_full_path_str}'. Skipping BigQuery write.")
                    except ValueError: 
                        st.error(f"Invalid BigQuery Table Path format. Expected 'project.dataset.table'. Error during parsing '{bq_full_path_str}'. Skipping BigQuery write.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during BigQuery operations setup or write: {e}")
                else:
                    st.warning("Master data is ready, but the BigQuery Table Path is missing. Skipping BigQuery write.")
            else:
                st.warning("Master data is empty or contains no valid records after processing. Skipping BigQuery write.")
        else:
            st.warning("Master data is empty (was not loaded or was cleared). Skipping BigQuery write.")
    else:
        st.info("No API data was fetched or processed successfully, so skipping BigQuery write.")
    return master_restaurant_data


# Create a button to trigger the API call
if st.button("Fetch Data"):
    handle_fetch_data_action(
        coordinate_pairs_str=coordinate_pairs_input,
        max_results=max_results_input,
        gcs_destination_uri_str=gcs_destination_uri,
        master_list_uri_str=master_list_uri,
        gcs_master_output_uri_str=gcs_master_dictionary_output_uri,
        bq_full_path_str=bq_full_path
    )
