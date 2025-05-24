# Standard Library
import json
import time
from datetime import datetime
from typing import List, Dict, Any # Simplified and reordered

# Third-party
import pandas as pd
import streamlit as st
from google.cloud import bigquery # For bigquery.SchemaField, kept `google.cloud` for consistency

# Local Modules
from api_client import fetch_api_data
from bq_utils import sanitize_column_name, write_to_bigquery
from data_processing import load_json_from_local_file_path, load_master_data, process_and_update_master_data
from gcs_utils import load_json_from_gcs, upload_to_gcs

# Removed comments about function definitions being moved previously

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
        time.sleep(4) # Pause for 4 seconds between API calls
        
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
    # Determine which loading function to use based on the URI scheme
    # Using master_list_uri_str as it's the parameter name in handle_fetch_data_action
    if master_list_uri_str.startswith("gs://"):
        load_function = load_json_from_gcs
    else:
        load_function = load_json_from_local_file_path
    
    master_restaurant_data = load_master_data(master_list_uri_str, load_function)

    # 3. Process API Response (Combined) and Update Master Restaurant Data
    master_restaurant_data, new_additions_count = process_and_update_master_data(master_restaurant_data, combined_api_data)
    # The success message from process_and_update_master_data already states new additions and total.

    # Prepare data for display filtering
    today_date_for_filtering = datetime.now().strftime("%Y-%m-%d")
    restaurants_to_display = []
    for restaurant in master_restaurant_data:
        if isinstance(restaurant, dict) and 'first_seen' in restaurant:
            if restaurant['first_seen'] == today_date_for_filtering:
                restaurants_to_display.append(restaurant)
    
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

        # Use the new upload_to_gcs that takes a dict
        if upload_to_gcs(data=api_data, destination_uri=full_gcs_path_api_response):
            st.success(f"Successfully uploaded combined raw API response to {full_gcs_path_api_response}")
        # upload_to_gcs from gcs_utils handles its own st.error messages on failure
    
    # 5. Upload Master Restaurant Data to gcs_master_output_uri_str (full file path)
    if gcs_master_output_uri_str:
        # Use the new upload_to_gcs that takes a dict
        if upload_to_gcs(data=master_restaurant_data, destination_uri=gcs_master_output_uri_str):
            st.success(f"Successfully uploaded master restaurant data to {gcs_master_output_uri_str}")
        # upload_to_gcs from gcs_utils handles its own st.error messages on failure

    # 6. Display Data
    display_data(master_restaurant_data)

    # 7. Write to BigQuery
    # The variable 'api_data' (holding combined_api_data) can be used to check if any data was fetched.
    if api_data and api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail'): 
        if master_restaurant_data: # Check if master data (potentially enriched) exists
            df_to_load = pd.json_normalize([item for item in master_restaurant_data if isinstance(item, dict)])

            # Ensure other potential date columns are handled if necessary,
            # for now, the issue is specifically with 'RatingDate'.
            # The 'first_seen' column is already a string in 'YYYY-MM-DD' format,
            # and its schema in BigQuery is 'DATE', which should be compatible.

            if df_to_load is not None and not df_to_load.empty:
                # Convert 'first_seen' to datetime objects if the column exists
                if 'first_seen' in df_to_load.columns:
                    df_to_load['first_seen'] = pd.to_datetime(df_to_load['first_seen'], errors='coerce')

                if bq_full_path_str:
                    try:
                        path_parts = bq_full_path_str.split('.')
                        if len(path_parts) == 3: 
                            project_id, dataset_id, table_id = path_parts
                            if len(project_id) > 0 and len(dataset_id) > 0 and len(table_id) > 0:
                                st.info(f"Attempting to write to BigQuery table: {bq_full_path_str}")
                                # Define columns to select and BigQuery schema
                                columns_to_select = [
                                    'FHRSID', 'BusinessName','AddressLine1', 'AddressLine2', 'AddressLine3', 'PostCode', 
                                    'RatingValue', 'RatingKey', 'RatingDate', 'LocalAuthorityName', 
                                    'Scores.Hygiene', 'Scores.Structural', 
                                    'Scores.ConfidenceInManagement', 'SchemeType', 'NewRatingPending', 
                                    'Geocode.Latitude', 'Geocode.Longitude', 'first_seen'
                                ]
                                # Filter out columns that are not in df_to_load to prevent errors
                                columns_to_select = [col for col in columns_to_select if col in df_to_load.columns]

                                bq_schema = [
                                    bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('AddressLine1'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('AddressLine2'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('AddressLine3'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('PostCode'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('RatingValue'), 'STRING'), # RatingValue can be non-numeric e.g. "Exempt"
                                    bigquery.SchemaField(sanitize_column_name('RatingKey'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('RatingDate'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('LocalAuthorityName'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('Scores.Hygiene'), 'INTEGER'),
                                    bigquery.SchemaField(sanitize_column_name('Scores.Structural'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('Scores.ConfidenceInManagement'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('Geocode.Longitude'), 'FLOAT'),
                                    bigquery.SchemaField(sanitize_column_name('Geocode.Latitude'), 'FLOAT'),
                                    bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE')
                                ]
                                # Filter schema to only include selected and sanitized columns
                                sanitized_columns_to_select_set = {sanitize_column_name(col) for col in columns_to_select}
                                bq_schema = [field for field in bq_schema if field.name in sanitized_columns_to_select_set]

                                # Convert 'Scores.Hygiene' to Int64
                                hygiene_col = 'Scores.Hygiene'
                                if hygiene_col in df_to_load.columns:
                                    st.write(f"Attempting to convert column: {hygiene_col}") # Added for debugging visibility
                                    df_to_load[hygiene_col] = pd.to_numeric(df_to_load[hygiene_col], errors='coerce')
                                    df_to_load[hygiene_col] = df_to_load[hygiene_col].astype('Int64')
                                    st.write(f"Conversion of {hygiene_col} complete. Dtype: {df_to_load[hygiene_col].dtype}") # Added for debugging

                                write_to_bigquery(df_to_load, project_id, dataset_id, table_id, columns_to_select, bq_schema)
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
