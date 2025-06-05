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
from bq_utils import sanitize_column_name, write_to_bigquery, read_from_bigquery
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

def fhrsid_lookup_logic(fhrsid_input_str: str, bq_table_lookup_input_str: str, st_object, read_from_bq_func, pd_concat_func):
    """
    Handles the logic for the FHRSID lookup.
    Args:
        fhrsid_input_str: The string input for FHRSIDs.
        bq_table_lookup_input_str: The string input for the BigQuery table.
        st_object: The streamlit object (or mock) for UI calls.
        read_from_bq_func: Function to call for reading from BigQuery.
        pd_concat_func: Function to call for concatenating pandas DataFrames (not strictly needed if read_from_bq_func returns combined df).
    """
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

        fhrsid_list_requested = [fhrsid.strip() for fhrsid in fhrsid_input_str.split(':') if fhrsid.strip()]
        if not fhrsid_list_requested:
            st_object.error("Please enter valid FHRSIDs.")
            return

        final_df = read_from_bq_func(fhrsid_list_requested, project_id, dataset_id, table_id)

        if final_df is not None and not final_df.empty:
            if 'fhrsid' in final_df.columns: # Ensure the key column exists
                successful_fhrsids_from_df = final_df['fhrsid'].astype(str).unique().tolist()
            else: # Should ideally not happen if data source is consistent
                successful_fhrsids_from_df = []
                st_object.error("FHRSID column ('fhrsid') missing in returned data. Cannot determine successful lookups.")

            if successful_fhrsids_from_df:
                st_object.success(f"Data found for FHRSIDs: {', '.join(successful_fhrsids_from_df)}")
                st_object.dataframe(final_df)

            failed_fhrsids = [f_id for f_id in fhrsid_list_requested if f_id not in successful_fhrsids_from_df]
            if failed_fhrsids:
                st_object.warning(f"No data found or error for FHRSIDs: {', '.join(failed_fhrsids)}")

            if not successful_fhrsids_from_df and not failed_fhrsids and not final_df.empty:
                 # This case means dataframe is not empty, but no FHRSIDs identified from it matched the requested list.
                 # This could happen if the 'fhrsid' column was present but empty or all values were different.
                 st_object.warning(f"Data returned but no matching FHRSIDs found for: {fhrsid_input_str} in {bq_table_lookup_input_str}.")
            elif not successful_fhrsids_from_df and not final_df.empty:
                 # Dataframe not empty, fhrsid column was missing, already handled by st.error above.
                 pass


        else: # final_df is None or empty
            st_object.warning(f"No data found for the provided FHRSIDs: {fhrsid_input_str} in {bq_table_lookup_input_str}, or an error occurred during lookup for all specified IDs.")
    except ValueError:
        st_object.error("Invalid BigQuery Table Path format. Expected 'project.dataset.table'.")
    except Exception as e:
        st_object.error(f"An unexpected error occurred during lookup: {e}")

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
        # Assuming max_results_input is accessible here, if not, it should be passed or defined globally.
        # For now, let's assume it's defined in the scope where handle_fetch_data_action is called (e.g. main_ui)
        # and passed appropriately if necessary. The test will mock it.
        # The error was that max_results_input was defined outside the if block, it should be inside if app_mode == "Fetch API Data"
        # This function is called from main_ui where max_results_input_ui is defined.
        api_response = fetch_api_data(lon, lat, max_results) # Using max_results param
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
    app_mode = st.radio("Choose an action:", ("Fetch API Data", "FHRSID Lookup"))

    if app_mode == "Fetch API Data":
        st.subheader("Fetch API Data and Update Master List")
        coordinate_pairs_input = st.text_area("Enter longitude,latitude pairs (one per line):")
        # These _ui variables are used to distinguish from the parameters of handle_fetch_data_action
        max_results_input_ui = st.number_input("Enter Max Results for API Call", min_value=1, max_value=5000, value=200)
        gcs_destination_uri_ui = st.text_input("Enter GCS destination folder for the scan (e.g., gs://bucket-name/scans-folder/)")
        master_list_uri_ui = st.text_input("Enter Master Restaurant Data URI (JSON) (e.g., gs://bucket/file.json or /path/to/file.json)")
        gcs_master_dictionary_output_uri_ui = st.text_input("Enter GCS URI for Master Restaurant Data Output (e.g., gs://bucket-name/path/filename.json)")
        bq_full_path_ui = st.text_input("Enter BigQuery Table Path (project.dataset.table)")

        if st.button("Fetch Data"):
            handle_fetch_data_action( # Pass the UI input values to the handler
                coordinate_pairs_str=coordinate_pairs_input,
                max_results=max_results_input_ui,
                gcs_destination_uri_str=gcs_destination_uri_ui,
                master_list_uri_str=master_list_uri_ui,
                gcs_master_output_uri_str=gcs_master_dictionary_output_uri_ui,
                bq_full_path_str=bq_full_path_ui
            )
    elif app_mode == "FHRSID Lookup":
        st.subheader("FHRSID Lookup")
        fhrsid_input_str_ui = st.text_input("Enter FHRSIDs (colon-separated):")
        bq_table_lookup_input_str_ui = st.text_input("Enter BigQuery Table Path for Lookup (project.dataset.table):")

        if st.button("Lookup FHRSIDs"):
            fhrsid_lookup_logic(fhrsid_input_str_ui, bq_table_lookup_input_str_ui, st, read_from_bigquery, pd.concat)

if __name__ == "__main__":
    main_ui()
