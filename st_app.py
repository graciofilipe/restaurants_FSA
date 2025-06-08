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

def fhrsid_lookup_logic(fhrsid_input_str: str, bq_table_lookup_input_str: str, st_object, read_from_bq_func): # Removed pd_concat_func
    """
    Handles the logic for the FHRSID lookup.
    Args:
        fhrsid_input_str: The string input for FHRSIDs.
        bq_table_lookup_input_str: The string input for the BigQuery table.
        st_object: The streamlit object (or mock) for UI calls.
        read_from_bq_func: Function to call for reading from BigQuery.
    """
    # Initialize or clear session state variables at the beginning of the logic
    st_object.session_state.fhrsid_df = pd.DataFrame() # Initialize with empty DataFrame
    st_object.session_state.successful_fhrsids = []

    if not bq_table_lookup_input_str:
        st_object.error("BigQuery Table Path is required.")
        return
    if not fhrsid_input_str:
        st_object.error("Please enter one or more FHRSIDs.")
        return

    try:
        project_id, dataset_id, table_id = bq_table_lookup_input_str.split('.')
        if not project_id or not dataset_id or not table_id:
            st_object.error("Invalid BigQuery Table Path format. Each part of 'project.dataset.table' must be non-empty.")
            return

        fhrsid_list_requested = [fhrsid.strip() for fhrsid in fhrsid_input_str.split(',') if fhrsid.strip()]
        fhrsid_list_requested = [f_id for f_id in fhrsid_list_requested if f_id] # Ensure no empty strings if input was like "123,,456"
        if not fhrsid_list_requested:
            st_object.error("Please enter valid FHRSIDs.")
            return

        # Validate FHRSIDs as numbers but store them as strings
        fhrsid_list_validated_strings = []
        for f_id_str in fhrsid_list_requested:
            try:
                int(f_id_str) # Validate that f_id_str is a number
                fhrsid_list_validated_strings.append(f_id_str) # Store the original string
            except ValueError:
                st_object.error(f"Invalid FHRSID: '{f_id_str}' is not a valid number. Please enter numeric FHRSIDs only.")
                return # Stop processing

        st_object.info(f"FHRSID Lookup: Attempting to retrieve data for {len(fhrsid_list_validated_strings)} FHRSID(s): {', '.join(fhrsid_list_requested)} in a single batch.")

        final_df = None # Explicitly initialize final_df to None
        try:
            # Call read_from_bq_func once with the entire list of validated strings
            # read_from_bigquery now returns an empty DataFrame if no records are found, or raises an error.
            final_df = read_from_bq_func(fhrsid_list_validated_strings, project_id, dataset_id, table_id)
        except BigQueryExecutionError as e: # Catching specific BQ execution errors
            st_object.error(f"BigQuery error during lookup for FHRSIDs {', '.join(fhrsid_list_requested)}: {e}") # Log original string list
            # DataFrameConversionError is less likely here as pandas-gbq handles conversion,
            # but if read_from_bq_func could still raise it, it would be caught by the generic Exception.
        except Exception as e: # Catch any other unexpected errors during the BQ call
            st_object.error(f"An unexpected error occurred during BigQuery lookup for FHRSIDs {', '.join(fhrsid_list_requested)}: {e}") # Log original string list

        if final_df is not None and not final_df.empty:
            if 'fhrsid' in final_df.columns:
                # Ensure fhrsid is string type for comparison, handle potential float if some are numbers
                successful_fhrsids_from_df = final_df['fhrsid'].astype(str).unique().tolist()
                st_object.session_state.successful_fhrsids = successful_fhrsids_from_df
                st_object.session_state.fhrsid_df = final_df # Store the dataframe
            else:
                st_object.error("FHRSID column ('fhrsid') missing in returned data. Cannot determine successful lookups.")

            if st_object.session_state.successful_fhrsids:
                st_object.success(f"Data found for FHRSIDs: {', '.join(st_object.session_state.successful_fhrsids)}")

            # Determine FHRSIDs for which no data was returned in the batch
            failed_fhrsids = [f_id for f_id in fhrsid_list_requested if f_id not in st_object.session_state.successful_fhrsids]
            if failed_fhrsids:
                st_object.warning(f"No data found for FHRSIDs: {', '.join(failed_fhrsids)} within the batch.")

        elif final_df is not None and final_df.empty: # Explicitly handle empty DataFrame case
             st_object.warning(f"No data found for any of the provided FHRSIDs: {', '.join(fhrsid_list_requested)}.")
        # If final_df is None, it means an error occurred and was handled above.
        # The session state for fhrsid_df is already an empty DataFrame.
        # successful_fhrsids is already an empty list.

    except ValueError: # This is for the bq_table_lookup_input_str.split('.')
        st_object.error("Invalid BigQuery Table Path format. Expected 'project.dataset.table'.")
        st_object.session_state.fhrsid_df = None
        st_object.session_state.successful_fhrsids = []
    except Exception as e:
        st_object.error(f"An unexpected error occurred during lookup: {e}")
        st_object.session_state.fhrsid_df = None
        st_object.session_state.successful_fhrsids = []


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
        st.stop()

    all_api_establishments = []

    st.info(f"Found {len(valid_coords)} valid coordinate pairs. Fetching data for each...")

    for lon, lat in valid_coords:
        st.write(f"Fetching data for Longitude: {lon}, Latitude: {lat}...")
        api_response = fetch_api_data(lon, lat, max_results)
        time.sleep(4)
        
        if api_response:
            establishments_list = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
            if establishments_list is None:
                establishments_list = []
            
            num_results_this_call = len(establishments_list)
            st.info(f"API call for ({lon}, {lat}) returned {num_results_this_call} establishments.")

            if num_results_this_call == max_results:
                st.warning(f"Warning for ({lon}, {lat}): The API returned {num_results_this_call} results, matching `max_results`. Results might be capped.")
            
            all_api_establishments.extend(establishments_list)

    if not all_api_establishments:
        st.info("No establishments found from any of the API calls. Nothing to process further.")
        st.stop()

    st.success(f"Total establishments fetched from all API calls: {len(all_api_establishments)}")
    
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}
    api_data = combined_api_data 

    if master_list_uri_str.startswith("gs://"):
        load_function = load_json_from_gcs
    else:
        load_function = load_json_from_local_file_path
    
    master_restaurant_data = load_master_data(master_list_uri_str, load_function)
    master_restaurant_data, _ = process_and_update_master_data(master_restaurant_data, combined_api_data)

    if gcs_destination_uri_str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        api_response_filename = f"combined_api_response_{current_date}.json" 
        gcs_destination_uri_folder = gcs_destination_uri_str
        if not gcs_destination_uri_folder.endswith('/'):
            gcs_destination_uri_folder += '/'
        full_gcs_path_api_response = f"{gcs_destination_uri_folder}{api_response_filename}"
        if upload_to_gcs(data=api_data, destination_uri=full_gcs_path_api_response):
            st.success(f"Successfully uploaded combined raw API response to {full_gcs_path_api_response}")
    
    if gcs_master_output_uri_str:
        if upload_to_gcs(data=master_restaurant_data, destination_uri=gcs_master_output_uri_str):
            st.success(f"Successfully uploaded master restaurant data to {gcs_master_output_uri_str}")

    display_data(master_restaurant_data)

    if api_data and api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail'): 
        if master_restaurant_data:
            df_to_load = pd.json_normalize([item for item in master_restaurant_data if isinstance(item, dict)])
            if df_to_load is not None and not df_to_load.empty:
                if 'first_seen' in df_to_load.columns:
                    df_to_load['first_seen'] = pd.to_datetime(df_to_load['first_seen'], errors='coerce')
                if bq_full_path_str:
                    try:
                        path_parts = bq_full_path_str.split('.')
                        if len(path_parts) == 3: 
                            project_id, dataset_id, table_id = path_parts
                            if len(project_id) > 0 and len(dataset_id) > 0 and len(table_id) > 0:
                                st.info(f"Attempting to write to BigQuery table: {bq_full_path_str}")
                                columns_to_select = [
                                    'FHRSID', 'BusinessName','AddressLine1', 'AddressLine2', 'AddressLine3', 'PostCode', 
                                    'RatingValue', 'RatingKey', 'RatingDate', 'LocalAuthorityName', 
                                    'Scores.Hygiene', 'Scores.Structural', 
                                    'Scores.ConfidenceInManagement', 'SchemeType', 'NewRatingPending', 
                                    'Geocode.Latitude', 'Geocode.Longitude', 'first_seen', 'manual_review'
                                ]
                                columns_to_select = [col for col in columns_to_select if col in df_to_load.columns]
                                bq_schema = [
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
                                    bigquery.SchemaField(sanitize_column_name('Scores.Hygiene'), 'INTEGER'),
                                    bigquery.SchemaField(sanitize_column_name('Scores.Structural'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('Scores.ConfidenceInManagement'), 'STRING'),
                                    bigquery.SchemaField(sanitize_column_name('Geocode.Longitude'), 'FLOAT'),
                                    bigquery.SchemaField(sanitize_column_name('Geocode.Latitude'), 'FLOAT'),
                                    bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE'),
                                    bigquery.SchemaField("manual_review", "STRING", mode="NULLABLE")
                                ]
                                sanitized_columns_to_select_set = {sanitize_column_name(col) for col in columns_to_select}
                                bq_schema = [field for field in bq_schema if field.name in sanitized_columns_to_select_set]
                                hygiene_col = 'Scores.Hygiene'
                                if hygiene_col in df_to_load.columns:
                                    st.write(f"Attempting to convert column: {hygiene_col}")
                                    df_to_load[hygiene_col] = pd.to_numeric(df_to_load[hygiene_col], errors='coerce')
                                    df_to_load[hygiene_col] = df_to_load[hygiene_col].astype('Int64')
                                    st.write(f"Conversion of {hygiene_col} complete. Dtype: {df_to_load[hygiene_col].dtype}")
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
                                    st.success(f"Manual review updated for FHRSIDs: {', '.join(selected_fhrsids_for_update)}. Refreshing data...")
                                    # Refresh data for all updated FHRSIDs
                                    # Clear old df from session state so UI elements dependent on it will refresh
                                    st.session_state.fhrsid_df = None
                                    # successful_fhrsids will be repopulated by fhrsid_lookup_logic

                                    # We need to re-lookup the data for the FHRSIDs that were attempted to be updated.
                                    # The original list of ALL FHRSIDs initially entered by the user is in st.session_state.fhrsid_input_str_ui
                                    # However, to show just the updated ones and any others previously successfully found,
                                    # it's better to re-query based on st.session_state.successful_fhrsids OR selected_fhrsids_for_update.
                                    # For simplicity and to ensure the view is consistent with what was just updated + other prior successes:
                                    # We will re-run the lookup for all *currently known successful* FHRSIDs if we want to keep them in view,
                                    # or just the selected_fhrsids_for_update if we only want to see the ones we just changed.
                                    # Let's re-lookup all previously successful FHRSIDs to maintain context.

                                    # Convert list of FHRSIDs to colon-separated string for fhrsid_lookup_logic
                                    fhrsids_to_refresh_str = ":".join(st.session_state.successful_fhrsids)

                                    fhrsid_lookup_logic(
                                        fhrsids_to_refresh_str,
                                        st.session_state.bq_table_lookup_input_str_ui,
                                        st,
                                        read_from_bigquery # Removed pd.concat
                                    )
                                    st.rerun()
                        except ValueError:
                            st.error("Invalid BigQuery Table Path format for update. Expected 'project.dataset.table'.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during update for FHRSIDs {', '.join(selected_fhrsids_for_update)}: {e}")
            elif st.session_state.successful_fhrsids: # successful_fhrsids exist, but none are selected for update
                st.info("Select one or more FHRSIDs from the list above to update their manual review status.")

        elif st.session_state.fhrsid_input_str_ui and st.button("Retry Lookup?", key="retry_lookup"): # Example of how to handle no data after attempt
             # This button is just an example, the main "Lookup FHRSIDs" serves this purpose
             st.write("Click 'Lookup FHRSIDs' again to retry.")


if __name__ == "__main__":
    main_ui()
