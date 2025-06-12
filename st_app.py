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
from bq_utils import (
    sanitize_column_name,
    write_to_bigquery,
    read_from_bigquery,
    update_manual_review,
    BigQueryExecutionError,  # Added import
    DataFrameConversionError # Added import
)
from data_processing import load_json_from_local_file_path, load_master_data, process_and_update_master_data
from gcs_utils import load_json_from_gcs, upload_to_gcs

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

        valid_items_for_df = [item for item in data_to_display if isinstance(item, dict)]
        
        if not valid_items_for_df:
            st.warning("Master restaurant data contains no dictionary items, cannot display as table.")
        elif len(valid_items_for_df) < len(data_to_display):
            st.warning(f"Some items in the master restaurant data were not dictionaries and were excluded from the table display. Displaying {len(valid_items_for_df)} records.")
        
        if valid_items_for_df:
            df = pd.json_normalize(valid_items_for_df)
            st.dataframe(df)
        
    except Exception as e: 
        st.error(f"Error displaying DataFrame from master restaurant data: {e}")
        st.info("Attempting to show raw master restaurant data as JSON.")
        try:
            st.json(data_to_display)
        except Exception as json_e:
            st.error(f"Could not even display master restaurant data as JSON: {json_e}")

def _validate_fhrsid_inputs(fhrsid_input_str: str, bq_table_lookup_input_str: str):
    """
    Validates FHRSID and BigQuery table string inputs.
    """
    if not bq_table_lookup_input_str:
        return None, None, None, None, "BigQuery Table Path is required."

    try:
        project_id, dataset_id, table_id = bq_table_lookup_input_str.split('.')
        if not project_id or not dataset_id or not table_id:
            return None, None, None, None, "Invalid BigQuery Table Path format. Each part of 'project.dataset.table' must be non-empty."
    except ValueError:
        return None, None, None, None, "Invalid BigQuery Table Path format. Expected 'project.dataset.table'."

    if not fhrsid_input_str:
        return None, None, None, None, "Please enter one or more FHRSIDs."

    fhrsid_list_requested = [fhrsid.strip() for fhrsid in fhrsid_input_str.split(',') if fhrsid.strip()]
    if not fhrsid_list_requested: # Handles cases like "," or ",," or empty string
        return None, None, None, None, "Please enter valid FHRSIDs. The list is empty after stripping whitespace."

    fhrsid_list_validated_strings = []
    for f_id_str in fhrsid_list_requested:
        try:
            int(f_id_str)  # Validate that f_id_str is a number
            fhrsid_list_validated_strings.append(f_id_str)  # Store the original string
        except ValueError:
            return None, None, None, None, f"Invalid FHRSID: '{f_id_str}' is not a valid number. Please enter numeric FHRSIDs only."

    if not fhrsid_list_validated_strings: # Should be caught by previous checks, but as a safeguard
        return None, None, None, None, "No valid FHRSIDs provided after validation."

    return project_id, dataset_id, table_id, fhrsid_list_validated_strings, None


def _fetch_and_process_fhrsid_data(fhrsid_list_validated_strings: List[str], project_id: str, dataset_id: str, table_id: str, read_from_bq_func):
    """
    Fetches data from BigQuery using the provided function and processes the result.
    """
    final_df = None
    successful_fhrsids_from_df = []
    error_message = None
    warning_message = None

    try:
        final_df = read_from_bq_func(fhrsid_list_validated_strings, project_id, dataset_id, table_id)

        if final_df is not None:
            if not final_df.empty: # Only check for columns if the DataFrame is not empty
                if 'fhrsid' not in final_df.columns:
                    warning_message = "FHRSID column ('fhrsid') missing in returned data. Cannot determine successful lookups or proceed with updates."
                else:
                    successful_fhrsids_from_df = final_df['fhrsid'].astype(str).unique().tolist()
            # If final_df is empty, successful_fhrsids_from_df remains [], and warning_message remains None from this block
        # If final_df is None (should not happen if read_from_bq_func adheres to its contract of returning empty df or raising error)
        # or if it's an empty DataFrame, successful_fhrsids_from_df remains empty, which is correct.

    except BigQueryExecutionError as e:
        error_message = f"BigQuery error during lookup for FHRSIDs {', '.join(fhrsid_list_validated_strings)}: {e}"
        return None, [], error_message # Return None for df, empty list for successful_ids
    except Exception as e:
        error_message = f"An unexpected error occurred during BigQuery lookup: {e}"
        return None, [], error_message # Return None for df, empty list for successful_ids

    return final_df, successful_fhrsids_from_df, error_message or warning_message


def fhrsid_lookup_logic(fhrsid_input_str: str, bq_table_lookup_input_str: str, st_object, read_from_bq_func):
    """
    Handles the logic for the FHRSID lookup using helper functions for validation and data fetching.
    Args:
        fhrsid_input_str: The string input for FHRSIDs.
        bq_table_lookup_input_str: The string input for the BigQuery table.
        st_object: The streamlit object (or mock) for UI calls.
        read_from_bq_func: Function to call for reading from BigQuery.
    """
    # Initialize or clear session state variables at the beginning of the logic
    st_object.session_state.fhrsid_df = pd.DataFrame()
    st_object.session_state.successful_fhrsids = []

    # Step 1: Validate inputs
    project_id, dataset_id, table_id, fhrsid_list_validated_strings, error_msg = \
        _validate_fhrsid_inputs(fhrsid_input_str, bq_table_lookup_input_str)

    if error_msg:
        st_object.error(error_msg)
        return

    # Keep a copy of the originally requested (and validated) FHRSIDs to determine failures later
    # _validate_fhrsid_inputs already filters out invalid format, so this list contains numerically valid fhrsid strings
    original_requested_fhrsids = list(fhrsid_list_validated_strings)


    st_object.info(f"FHRSID Lookup: Attempting to retrieve data for {len(fhrsid_list_validated_strings)} FHRSID(s): {', '.join(fhrsid_list_validated_strings)} in a single batch.")

    # Step 2: Fetch and process data
    final_df, successful_fhrsids_from_df, message = \
        _fetch_and_process_fhrsid_data(fhrsid_list_validated_strings, project_id, dataset_id, table_id, read_from_bq_func)

    if message and "error" in message.lower(): # Check if the message is an error message
        st_object.error(message)
        # final_df might be None if a BQ error occurred, or it might be a df with missing 'fhrsid' column.
        # Session state for df is already an empty DataFrame. successful_fhrsids is an empty list.
        # No further processing needed if there was a critical error.
        return
    elif message: # This would be a warning (e.g. fhrsid column missing)
        st_object.warning(message)
        # If 'fhrsid' column is missing, successful_fhrsids_from_df will be empty.
        # We might still have a df in final_df, so we update session state.
        # It will be an empty dataframe if 'fhrsid' column is missing.

    st_object.session_state.fhrsid_df = final_df if final_df is not None else pd.DataFrame()
    st_object.session_state.successful_fhrsids = successful_fhrsids_from_df


    # Step 3: Display messages based on results
    if successful_fhrsids_from_df:
        st_object.success(f"Data found for FHRSIDs: {', '.join(successful_fhrsids_from_df)}")

    # Determine FHRSIDs for which no data was returned
    # Use the original_requested_fhrsids which passed initial validation
    failed_fhrsids = [f_id for f_id in original_requested_fhrsids if f_id not in successful_fhrsids_from_df]

    if final_df is not None and final_df.empty and not successful_fhrsids_from_df and not message:
        # This is the most specific case: query ran, returned no data for any requested ID.
        st_object.warning(f"No data found for any of the provided FHRSIDs: {', '.join(original_requested_fhrsids)}.")
    elif failed_fhrsids:
        # This case handles when some FHRSIDs failed, but not all, or if 'fhrsid' column was missing (making successful_fhrsids_from_df empty).
        st_object.warning(f"No data found for some of the provided FHRSIDs: {', '.join(failed_fhrsids)}.")

    # If final_df is None and an error occurred, it was handled at the beginning of Step 3.
    # If final_df is not None, it's stored in session_state. If it's empty, it's an empty DF.
    # successful_fhrsids is also updated in session_state.

# Helper functions for handle_fetch_data_action
def _parse_coordinates(coordinate_pairs_str: str) -> List[tuple[float, float]]:
    """
    Parses a string of coordinate pairs into a list of (lon, lat) tuples.
    Logs errors for invalid lines via st.error.
    """
    valid_coords = []
    coordinate_lines = coordinate_pairs_str.strip().split('\n')
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
            st.error(f"Error parsing coordinate line {i+1}: '{line}'. Expected 'longitude,latitude'. Skipping this line.")
            continue
    return valid_coords

def _fetch_data_for_all_coordinates(valid_coords: List[tuple[float, float]], max_results: int) -> List[Dict[str, Any]]:
    """
    Fetches API data for all valid coordinates and aggregates the results.
    Displays warnings if results might be capped.
    """
    all_api_establishments = []
    st.info(f"Found {len(valid_coords)} valid coordinate pairs. Fetching data for each...")
    for lon, lat in valid_coords:
        st.write(f"Fetching data for Longitude: {lon}, Latitude: {lat}...")
        api_response = fetch_api_data(lon, lat, max_results)
        time.sleep(4)  # Respect API rate limits
        
        if api_response:
            establishments_list = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
            if establishments_list is None:  # Handle cases where EstablishmentDetail might be null
                establishments_list = []
            
            num_results_this_call = len(establishments_list)
            st.info(f"API call for ({lon}, {lat}) returned {num_results_this_call} establishments.")

            if num_results_this_call == max_results:
                st.warning(f"Warning for ({lon}, {lat}): The API returned {num_results_this_call} results, matching `max_results`. Results might be capped.")
            
            all_api_establishments.extend(establishments_list)
    return all_api_establishments

def _handle_gcs_uploads(api_data: Dict[str, Any], master_restaurant_data: List[Dict[str, Any]], gcs_destination_uri_str: str, gcs_master_output_uri_str: str):
    """
    Handles uploading API response and master restaurant data to GCS.
    """
    if gcs_destination_uri_str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        api_response_filename = f"combined_api_response_{current_date}.json"
        gcs_destination_uri_folder = gcs_destination_uri_str
        if not gcs_destination_uri_folder.endswith('/'):
            gcs_destination_uri_folder += '/'
        full_gcs_path_api_response = f"{gcs_destination_uri_folder}{api_response_filename}"
        if upload_to_gcs(data=api_data, destination_uri=full_gcs_path_api_response):
            st.success(f"Successfully uploaded combined raw API response to {full_gcs_path_api_response}")
        # upload_to_gcs internally handles and logs errors if upload fails
    
    if gcs_master_output_uri_str:
        if upload_to_gcs(data=master_restaurant_data, destination_uri=gcs_master_output_uri_str):
            st.success(f"Successfully uploaded master restaurant data to {gcs_master_output_uri_str}")
        # upload_to_gcs internally handles and logs errors if upload fails

def _write_data_to_bigquery(master_restaurant_data: List[Dict[str, Any]], bq_full_path_str: str):
    """
    Prepares and writes master restaurant data to BigQuery.
    """
    if not master_restaurant_data:
        st.warning("Master data is empty. Skipping BigQuery write.")
        return

    df_to_load = pd.json_normalize([item for item in master_restaurant_data if isinstance(item, dict)])

    if df_to_load.empty:
        st.warning("Master data contains no valid dictionary records after normalization. Skipping BigQuery write.")
        return

    if 'first_seen' in df_to_load.columns:
        df_to_load['first_seen'] = pd.to_datetime(df_to_load['first_seen'], errors='coerce')

    if not bq_full_path_str:
        st.warning("BigQuery Table Path is missing. Skipping BigQuery write.")
        return

    try:
        path_parts = bq_full_path_str.split('.')
        if len(path_parts) != 3 or not all(path_parts): # Check for 3 non-empty parts
            st.error(f"Invalid BigQuery Table Path format: '{bq_full_path_str}'. Expected 'project.dataset.table' with non-empty parts. Skipping BigQuery write.")
            return
        project_id, dataset_id, table_id = path_parts

        st.info(f"Attempting to write to BigQuery table: {bq_full_path_str}")
        columns_to_select = [
            'FHRSID', 'BusinessName','AddressLine1', 'AddressLine2', 'AddressLine3', 'PostCode',
            'RatingValue', 'RatingKey', 'RatingDate', 'LocalAuthorityName',
            'Scores.Hygiene', 'Scores.Structural',
            'Scores.ConfidenceInManagement', 'SchemeType', 'NewRatingPending',
            'Geocode.Latitude', 'Geocode.Longitude', 'first_seen', 'manual_review', 'gemini_insights'
        ]
        # Filter columns_to_select to only those present in df_to_load
        columns_to_select = [col for col in columns_to_select if col in df_to_load.columns]
        if not columns_to_select:
            st.warning("No relevant columns found in the data for BigQuery. Skipping BigQuery write.")
            return

        # Define schema based on available columns
        bq_schema_all_possible = [
            bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('AddressLine1'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('AddressLine2'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('AddressLine3'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('PostCode'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('RatingValue'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('RatingKey'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('RatingDate'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('LocalAuthorityName'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('Scores.Hygiene'), 'INTEGER'), # Will be converted
            bigquery.SchemaField(sanitize_column_name('Scores.Structural'), 'STRING'), # Assuming string, adjust if numeric
            bigquery.SchemaField(sanitize_column_name('Scores.ConfidenceInManagement'), 'STRING'), # Assuming string
            bigquery.SchemaField(sanitize_column_name('SchemeType'), 'STRING'),
            bigquery.SchemaField(sanitize_column_name('NewRatingPending'), 'BOOLEAN'), # Assuming boolean, adjust if string
            bigquery.SchemaField(sanitize_column_name('Geocode.Longitude'), 'FLOAT'),
            bigquery.SchemaField(sanitize_column_name('Geocode.Latitude'), 'FLOAT'),
            bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE'),
            bigquery.SchemaField("manual_review", "STRING", mode="NULLABLE"),
            bigquery.SchemaField(sanitize_column_name('gemini_insights'), 'STRING', mode='NULLABLE')
        ]

        sanitized_columns_present_in_df = {sanitize_column_name(col) for col in columns_to_select}
        bq_schema = [field for field in bq_schema_all_possible if field.name in sanitized_columns_present_in_df]

        hygiene_col = 'Scores.Hygiene'
        if hygiene_col in df_to_load.columns:
            # st.write(f"Attempting to convert column: {hygiene_col}") # Usually too verbose for helper
            df_to_load[hygiene_col] = pd.to_numeric(df_to_load[hygiene_col], errors='coerce').astype('Int64')
            # st.write(f"Conversion of {hygiene_col} complete. Dtype: {df_to_load[hygiene_col].dtype}")

        # Note: write_to_bigquery from bq_utils handles sanitization of df column names before writing
        write_to_bigquery(df_to_load, project_id, dataset_id, table_id, columns_to_select, bq_schema)

    except ValueError: # Catch error from bq_full_path_str.split('.') if it wasn't 3 parts.
        st.error(f"Invalid BigQuery Table Path format during parsing: '{bq_full_path_str}'. Expected 'project.dataset.table'. Skipping BigQuery write.")
    except Exception as e:
        st.error(f"An unexpected error occurred during BigQuery operations setup or write: {e}")


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
    when the 'Fetch Data' button is clicked, using helper functions.
    """
    # 1. Parse and validate coordinates
    valid_coords = _parse_coordinates(coordinate_pairs_str)
    if not valid_coords:
        st.error("No valid coordinate pairs found. Please enter coordinates in 'longitude,latitude' format, one per line.")
        st.stop()

    # 2. Iterative API data fetching
    all_api_establishments = _fetch_data_for_all_coordinates(valid_coords, max_results)
    if not all_api_establishments:
        st.info("No establishments found from any of the API calls. Nothing to process further.")
        st.stop()
    st.success(f"Total establishments fetched from all API calls: {len(all_api_establishments)}")

    # Prepare combined API data structure (as expected by downstream functions)
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}

    # 3. Load master data
    if master_list_uri_str.startswith("gs://"):
        load_function = load_json_from_gcs
    else:
        load_function = load_json_from_local_file_path
    master_restaurant_data = load_master_data(master_list_uri_str, load_function)

    # 4. Process API data with master data
    master_restaurant_data, _ = process_and_update_master_data(master_restaurant_data, combined_api_data)

    # 5. Handle GCS Uploads
    _handle_gcs_uploads(combined_api_data, master_restaurant_data, gcs_destination_uri_str, gcs_master_output_uri_str)

    # 6. Display data
    display_data(master_restaurant_data)

    # 7. Write data to BigQuery (conditionally, based on api_data having results)
    if all_api_establishments: # Check if any data was fetched to justify writing
        _write_data_to_bigquery(master_restaurant_data, bq_full_path_str)
    else:
        st.info("No new API data was fetched, so skipping BigQuery write.") # Should be caught earlier, but as a safeguard

    return master_restaurant_data

def main_ui():
    st.title("Food Standards Agency API Explorer")

    # Initialize session state variables if they don't exist
    if 'fhrsid_df' not in st.session_state:
        st.session_state.fhrsid_df = pd.DataFrame() # Initialize with empty DataFrame
    if 'successful_fhrsids' not in st.session_state:
        st.session_state.successful_fhrsids = []
    if 'fhrsid_input_str_ui' not in st.session_state: # Persist input values
        st.session_state.fhrsid_input_str_ui = ""
    if 'bq_table_lookup_input_str_ui' not in st.session_state:
        st.session_state.bq_table_lookup_input_str_ui = ""


    app_mode = st.radio("Choose an action:", ("Fetch API Data", "FHRSID Lookup"))

    if app_mode == "Fetch API Data":
        # Reset FHRSID lookup session state if switching modes
        st.session_state.fhrsid_df = None
        st.session_state.successful_fhrsids = []
        st.subheader("Fetch API Data and Update Master List")
        coordinate_pairs_input = st.text_area("Enter longitude,latitude pairs (one per line):")
        # These _ui variables are used to distinguish from the parameters of handle_fetch_data_action
        max_results_input_ui = st.number_input("Enter Max Results for API Call", min_value=1, max_value=5000, value=200)
        gcs_destination_uri_ui = st.text_input("Enter GCS destination folder for the scan (e.g., gs://bucket-name/scans-folder/)")
        master_list_uri_ui = st.text_input("Enter Master Restaurant Data URI (JSON) (e.g., gs://bucket/file.json or /path/to/file.json)")
        gcs_master_dictionary_output_uri_ui = st.text_input("Enter GCS URI for Master Restaurant Data Output (e.g., gs://bucket-name/path/filename.json)")
        bq_full_path_ui = st.text_input("Enter BigQuery Table Path (project.dataset.table)")

        if st.button("Fetch Data"):
            handle_fetch_data_action(
                coordinate_pairs_str=coordinate_pairs_input,
                max_results=max_results_input_ui,
                gcs_destination_uri_str=gcs_destination_uri_ui,
                master_list_uri_str=master_list_uri_ui,
                gcs_master_output_uri_str=gcs_master_dictionary_output_uri_ui,
                bq_full_path_str=bq_full_path_ui
            )
    elif app_mode == "FHRSID Lookup":
        st.subheader("FHRSID Lookup")
        # Use session state to retain input values across reruns
        st.session_state.fhrsid_input_str_ui = st.text_input(
            "Enter FHRSIDs (comma-separated):",
            value=st.session_state.fhrsid_input_str_ui
        )
        st.session_state.bq_table_lookup_input_str_ui = st.text_input(
            "Enter BigQuery Table Path for Lookup (project.dataset.table):",
            value=st.session_state.bq_table_lookup_input_str_ui
        )

        if st.button("Lookup FHRSIDs"):
            # Call the logic function which now modifies session_state
            fhrsid_lookup_logic(
                st.session_state.fhrsid_input_str_ui,
                st.session_state.bq_table_lookup_input_str_ui,
                st,
                read_from_bigquery # Removed pd.concat
            )

        # Display data and update UI based on session_state populated by fhrsid_lookup_logic
        if st.session_state.fhrsid_df is not None and not st.session_state.fhrsid_df.empty:
            st.dataframe(st.session_state.fhrsid_df)

            st.subheader("Update Manual Review")

            selected_fhrsids_for_update = []
            if st.session_state.successful_fhrsids:
                # Use multiselect for selecting FHRSIDs
                selected_fhrsids_for_update = st.multiselect(
                    "Select FHRSID(s) to update:",
                    options=st.session_state.successful_fhrsids,
                    default=st.session_state.successful_fhrsids if len(st.session_state.successful_fhrsids) == 1 else [],
                    key="fhrsid_multiselect_update"
                )

            if selected_fhrsids_for_update: # Check if list is not empty
                # Create a unique key for the text input based on the selected FHRSIDs to avoid state issues
                # Using a hash of the sorted list of FHRSIDs for a stable key
                selected_ids_key_suffix = "_".join(sorted(selected_fhrsids_for_update))
                manual_review_input_ui = st.text_input(
                    "New Manual Review Value:",
                    key=f"manual_review_input_{selected_ids_key_suffix}"
                )

                if st.button("Update Manual Review"):
                    if not st.session_state.bq_table_lookup_input_str_ui:
                        st.error("BigQuery Table Path is required for update.")
                    elif not manual_review_input_ui.strip():
                        st.warning("Manual Review Value cannot be empty.")
                    else:
                        try:
                            project_id, dataset_id, table_id = st.session_state.bq_table_lookup_input_str_ui.split('.')
                            if not project_id or not dataset_id or not table_id:
                                st.error("Invalid BigQuery Table Path format for update. Each part must be non-empty.")
                            else:
                                # Call the updated update_manual_review function
                                update_successful = update_manual_review(
                                    fhrsid_list=selected_fhrsids_for_update, # Pass the list of selected FHRSIDs
                                    manual_review_value=manual_review_input_ui,
                                    project_id=project_id,
                                    dataset_id=dataset_id,
                                    table_id=table_id
                                )
                                if update_successful:
                                    st.success(f"Manual review updated for FHRSIDs: {', '.join(selected_fhrsids_for_update)} in BigQuery.")
                                    # Option A: Direct DataFrame Update
                                    # This approach updates the DataFrame in session state directly,
                                    # avoiding a re-query to BigQuery for a more responsive UI.
                                    if 'manual_review' not in st.session_state.fhrsid_df.columns:
                                        # If 'manual_review' column doesn't exist, add it with a default value (e.g., None or empty string)
                                        # This case should ideally not happen if data is properly formed.
                                        st.session_state.fhrsid_df['manual_review'] = pd.NA # or ""

                                    # Ensure FHRSID column is string type for matching, though it should be already
                                    st.session_state.fhrsid_df['fhrsid'] = st.session_state.fhrsid_df['fhrsid'].astype(str)

                                    # Update the 'manual_review' column for the selected FHRSIDs
                                    mask = st.session_state.fhrsid_df['fhrsid'].isin(selected_fhrsids_for_update)
                                    st.session_state.fhrsid_df.loc[mask, 'manual_review'] = manual_review_input_ui

                                    st.info("Local data view updated. Use 'Lookup FHRSIDs' again if you need to refresh from BigQuery.")
                                    # st.rerun() might still be needed if direct assignment doesn't refresh composed widgets.
                                    # For st.dataframe, direct modification of session_state bound data usually triggers a refresh.
                                    # If other dependent widgets don't update, uncomment st.rerun().
                                    # For now, we assume st.dataframe will pick up the change.
                                    # st.rerun() # Uncomment if UI does not refresh as expected.
                                else:
                                    # If update_successful is False, update_manual_review in bq_utils should have already shown an error.
                                    # No specific action here, as the error is handled by the called function.
                                    pass
                        except ValueError:
                            st.error("Invalid BigQuery Table Path format for update. Expected 'project.dataset.table'.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during update for FHRSIDs {', '.join(selected_fhrsids_for_update)}: {e}")
            elif st.session_state.successful_fhrsids: # successful_fhrsids exist, but none are selected for update
                st.info("Select one or more FHRSIDs from the list above to update their manual review status.")

        elif st.session_state.fhrsid_input_str_ui and not st.session_state.fhrsid_df.empty and st.button("Refresh Data from BigQuery", key="refresh_lookup"):
            # Added a button to explicitly re-fetch if needed, using the original input string for FHRSIDs
            st.info("Refreshing data from BigQuery...")
            fhrsid_lookup_logic(
                st.session_state.fhrsid_input_str_ui, # Use the original full input string
                st.session_state.bq_table_lookup_input_str_ui,
                st,
                read_from_bigquery
            )
            st.rerun()
        elif st.session_state.fhrsid_input_str_ui and st.session_state.fhrsid_df.empty and st.button("Retry Lookup?", key="retry_lookup"): # Existing retry
             # This button is just an example, the main "Lookup FHRSIDs" serves this purpose
             st.write("Click 'Lookup FHRSIDs' again to retry.")


if __name__ == "__main__":
    main_ui()
