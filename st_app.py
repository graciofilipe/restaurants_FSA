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
    load_all_data_from_bq, # Added import
    append_to_bigquery, # Added for new flow
    BigQueryExecutionError,  # Added import
    DataFrameConversionError, # Added import
    get_recent_restaurants,
    update_gemini_and_review_in_bq
)
from data_processing import load_json_from_local_file_path, load_master_data, process_and_update_master_data
from recent_restaurant_analysis import call_gemini_with_fhrs_data, create_recent_restaurants_temp_table

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

def _append_new_data_to_bigquery(new_restaurants: List[Dict[str, Any]], project_id: str, dataset_id: str, table_id: str):
    """
    Converts new restaurant data to a DataFrame, defines schema, sanitizes columns,
    and appends to the specified BigQuery table.
    """
    if not new_restaurants:
        st.info("No new restaurants found to add to BigQuery.")
        return

    st.info(f"Preparing {len(new_restaurants)} new records for BigQuery append...")
    df_new_restaurants = pd.json_normalize(new_restaurants)

    # Define comprehensive schema for append operation
    # Ensures all expected columns from new_restaurants are included and correctly typed.
    bq_schema_for_append = [
        bigquery.SchemaField(sanitize_column_name('FHRSID'), 'STRING'), # FHRSID is typically integer
        bigquery.SchemaField(sanitize_column_name('LocalAuthorityBusinessID'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('BusinessName'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('BusinessType'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('BusinessTypeID'), 'INTEGER'),
        bigquery.SchemaField(sanitize_column_name('AddressLine1'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('AddressLine2'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('AddressLine3'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('AddressLine4'), 'STRING', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('PostCode'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('RatingValue'), 'STRING'), # Can be "Pass", "Exempt", or number
        bigquery.SchemaField(sanitize_column_name('RatingKey'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('RatingDate'), 'STRING'), # Or DATE if time component is not important
        bigquery.SchemaField(sanitize_column_name('LocalAuthorityCode'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('LocalAuthorityName'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('LocalAuthorityWebSite'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('LocalAuthorityEmailAddress'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('Scores.Hygiene'), 'INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('Scores.Structural'), 'INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('Scores.ConfidenceInManagement'), 'INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(sanitize_column_name('SchemeType'), 'STRING'),
        bigquery.SchemaField(sanitize_column_name('Geocode.Longitude'), 'FLOAT'), # Handled by append_to_bigquery
        bigquery.SchemaField(sanitize_column_name('Geocode.Latitude'), 'FLOAT'),  # Handled by append_to_bigquery
        bigquery.SchemaField(sanitize_column_name('NewRatingPending'), 'BOOLEAN'), # Handled by append_to_bigquery
        bigquery.SchemaField(sanitize_column_name('first_seen'), 'DATE'),
        bigquery.SchemaField(sanitize_column_name('manual_review'), 'STRING')
    ]

    # Sanitize DataFrame column names to match BQ schema conventions
    # And ensure only columns defined in the schema are present, in the correct order.
    sanitized_df_columns = {}
    final_df_cols_for_bq = []
    schema_field_names = [field.name for field in bq_schema_for_append]

    for orig_col in df_new_restaurants.columns:
        sanitized = sanitize_column_name(orig_col)
        sanitized_df_columns[orig_col] = sanitized
        if sanitized in schema_field_names:
            final_df_cols_for_bq.append(sanitized)

    df_new_restaurants.columns = df_new_restaurants.columns.map(sanitized_df_columns)

    # Data type conversions for specific columns before sending to BQ
    # append_to_bigquery handles Geocode and NewRatingPending

    s_first_seen = sanitize_column_name('first_seen')
    if s_first_seen in df_new_restaurants.columns:
        df_new_restaurants[s_first_seen] = pd.to_datetime(df_new_restaurants[s_first_seen]).dt.date
    else:
        st.warning(f"Column '{s_first_seen}' (expected for first_seen date) not found in new restaurants DataFrame. It will be skipped for BQ append.")


    s_rating_date = sanitize_column_name('RatingDate')
    if s_rating_date in df_new_restaurants.columns:
        # Ensure the column is treated as string.
        # If it was parsed as datetime, convert to ISO string.
        # If it's already string, this should be mostly idempotent.
        # If it's None/NaT, it should become None.
        def stringify_date(x):
            if pd.isna(x):
                return None
            if isinstance(x, str):
                return x
            try:
                # Attempt to format if it's a datetime-like object
                return pd.Timestamp(x).strftime('%Y-%m-%d')
            except ValueError:
                # If conversion to Timestamp fails, just convert to string
                return str(x)

        df_new_restaurants[s_rating_date] = df_new_restaurants[s_rating_date].apply(stringify_date)
        # Ensure the final dtype is object (which Pandas uses for strings) to be safe.
        df_new_restaurants[s_rating_date] = df_new_restaurants[s_rating_date].astype('object')
    else:
        st.warning(f"Column '{s_rating_date}' (expected for RatingDate) not found. It will be skipped for BQ append.")


    score_cols_original = ['Scores.Hygiene', 'Scores.Structural', 'Scores.ConfidenceInManagement']
    for orig_col_name in score_cols_original:
        s_col_name = sanitize_column_name(orig_col_name)
        if s_col_name in df_new_restaurants.columns:
            df_new_restaurants[s_col_name] = pd.to_numeric(df_new_restaurants[s_col_name], errors='coerce').astype('Int64') # Use Int64 for nullable integers
        else:
            # It's possible scores are not present for all records, so this might not be a warning if optional
            print(f"Info: Score column '{s_col_name}' not found in new restaurants DataFrame. Will be skipped if not in schema or be Null.")

    # Ensure FHRSID is string
    s_fhrsid = sanitize_column_name('FHRSID')
    if s_fhrsid in df_new_restaurants.columns:
        # Convert all values in the column to string, including None or NA
        df_new_restaurants[s_fhrsid] = df_new_restaurants[s_fhrsid].astype(str)

    s_business_type_id = sanitize_column_name('BusinessTypeID')
    if s_business_type_id in df_new_restaurants.columns:
        df_new_restaurants[s_business_type_id] = pd.to_numeric(df_new_restaurants[s_business_type_id], errors='coerce').astype('Int64')

    s_lon = sanitize_column_name('Geocode.Longitude')
    if s_lon in df_new_restaurants.columns:
        df_new_restaurants[s_lon] = pd.to_numeric(df_new_restaurants[s_lon], errors='coerce')

    s_lat = sanitize_column_name('Geocode.Latitude')
    if s_lat in df_new_restaurants.columns:
        df_new_restaurants[s_lat] = pd.to_numeric(df_new_restaurants[s_lat], errors='coerce')

    # Reorder df_new_restaurants columns to match schema order and select only schema columns
    # This is important because append_to_bigquery itself will select based on schema, but good practice.
    # df_new_restaurants should now have sanitized column names
    cols_to_keep = [field.name for field in bq_schema_for_append if field.name in df_new_restaurants.columns]
    df_for_bq = df_new_restaurants[cols_to_keep]

    # Filter schema to only include fields present in the DataFrame to avoid errors
    # if some fields (e.g. a score field) were missing from all new_restaurants records
    final_bq_schema = [field for field in bq_schema_for_append if field.name in df_for_bq.columns]

    if df_for_bq.empty:
        st.warning("After processing, the DataFrame for new restaurants is empty. Skipping BigQuery append.")
        return

    success = append_to_bigquery(
        df=df_for_bq,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        bq_schema=final_bq_schema
    )

    if success:
        st.success(f"Successfully appended {len(df_for_bq)} new records to BigQuery table {project_id}.{dataset_id}.{table_id}.")
    else:
        st.error(f"Failed to append new records to BigQuery table {project_id}.{dataset_id}.{table_id}.")


def handle_fetch_data_action(
    coordinate_pairs_str: str,
    max_results: int,
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
        return []

    # Validate bq_full_path_str EARLY for master data loading
    if not bq_full_path_str:
        st.error("BigQuery Table Path (for master data and writing) is missing.")
        st.stop()
        return []
    try:
        project_id_master_val, dataset_id_master_val, table_id_master_val = bq_full_path_str.split('.')
        if not project_id_master_val or not dataset_id_master_val or not table_id_master_val:
            st.error("Invalid BigQuery Table Path format. Each part of 'project.dataset.table' must be non-empty.")
            st.stop()
            return []
    except ValueError:
        st.error(f"Invalid BigQuery Table Path format: '{bq_full_path_str}'. Expected 'project.dataset.table'.")
        st.stop()
        return []

    # 2. Iterative API data fetching
    all_api_establishments = _fetch_data_for_all_coordinates(valid_coords, max_results)
    if not all_api_establishments:
        st.info("No establishments found from any of the API calls. Nothing to process further.")
        st.stop()
        # Return [] removed here to allow master data loading as per test expectation
    st.success(f"Total establishments fetched from all API calls: {len(all_api_establishments)}")

    # Prepare combined API data structure (as expected by downstream functions)
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}

    # 3. Load master data
    master_restaurant_data = [] # Initialize as empty list
    # Validation and parsing of project_id_master_val, dataset_id_master_val, table_id_master_val
    # from bq_full_path_str has already been done before API calls.

    # 3a. Load master data using load_master_data and passing load_all_data_from_bq
    master_restaurant_data = load_master_data(
        project_id=project_id_master_val,
        dataset_id=dataset_id_master_val,
        table_id=table_id_master_val,
        load_bq_func=load_all_data_from_bq # Pass the actual function
    )
    # load_master_data handles logging and returns [] on error or if empty.

    # 4. Process API data with master data to find new restaurants
    # master_restaurant_data (from BQ) is used to check for existing FHRSIDs.
    # new_restaurants will be a list of dicts for new items.
    new_restaurants = process_and_update_master_data(master_restaurant_data, combined_api_data)
    # process_and_update_master_data handles its own logging for new records identified.

    # 5. Append new restaurants to BigQuery
    # The bq_full_path_str is for the same master table, which we are appending to.
    if not bq_full_path_str:
        st.warning("BigQuery Table Path for append is missing. Skipping BigQuery append for new records.")
    else:
        try:
            project_id_append, dataset_id_append, table_id_append = bq_full_path_str.split('.')
            if not all([project_id_append, dataset_id_append, table_id_append]):
                st.error(f"Invalid BigQuery Table Path for append: '{bq_full_path_str}'. Each part must be non-empty.")
            else:
                # Call the new helper to handle the append logic
                _append_new_data_to_bigquery(new_restaurants, project_id_append, dataset_id_append, table_id_append)
        except ValueError:
            st.error(f"Invalid BigQuery Table Path format for append: '{bq_full_path_str}'. Expected 'project.dataset.table'.")
        except Exception as e: # Catch any other unexpected error during append setup
            st.error(f"An unexpected error occurred before appending to BigQuery: {e}")


    # 6. Handle GCS Uploads
    # The master_restaurant_data here is the initial load from BQ.
    # If gcs_master_output_uri_str was meant to store the *appended* state, this needs adjustment.
    # For now, sticking to the instruction to leave as is.

    # 7. Display data
    # This displays the initial master_restaurant_data. If the display should reflect the appended data,
    # it would need to either re-load from BQ or combine master_restaurant_data + new_restaurants (if df versions are compatible).
    # For now, sticking to displaying initial load.
    st.info("Displaying master data loaded from BigQuery (before current API fetch append).") # Modified the info message slightly
    display_data(master_restaurant_data) # Explicitly display the loaded master data

    # If new_restaurants were found, also display them for clarity in this run
    if new_restaurants:
        st.subheader(f"Newly identified restaurants from this fetch ({len(new_restaurants)}):")
        display_data(new_restaurants) # Use the same display_data function

    # The function is expected to return List[Dict[str, Any]].
    # Returning the initial master_data for now, as the old _write_data_to_bigquery
    # was also based on the (then modified) master_data.
    # If the expectation is to return the "complete" data after append, this would need adjustment.
    return master_restaurant_data

def main_ui():
    st.title("Food Standards Agency API Explorer")

    # Initialize session state variables if they don't exist
    if 'recent_restaurants_df' not in st.session_state:
        st.session_state.recent_restaurants_df = None
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = None
    if 'current_dataset_id' not in st.session_state:
        st.session_state.current_dataset_id = None


    app_mode = st.radio("Choose an action:", ("Fetch API Data", "Recent Restaurant Analysis"))

    if app_mode == "Fetch API Data":
        st.subheader("Fetch API Data and Update Master List")
        coordinate_pairs_input = st.text_area("Enter longitude,latitude pairs (one per line):")
        # These _ui variables are used to distinguish from the parameters of handle_fetch_data_action
        max_results_input_ui = st.number_input("Enter Max Results for API Call", min_value=1, max_value=5000, value=200)
        bq_full_path_ui = st.text_input("Enter BigQuery Table Path for master data and to write updated data (project.dataset.table)")

        if st.button("Fetch Data"):
            handle_fetch_data_action(
                coordinate_pairs_str=coordinate_pairs_input,
                max_results=max_results_input_ui,
                bq_full_path_str=bq_full_path_ui
            )
    elif app_mode == "Recent Restaurant Analysis":
        st.subheader("Analyze Recently Added Restaurants")

        # Input for N_DAYS
        n_days_input = st.number_input(
            "Enter the number of days to look back for recent restaurants (N_DAYS):",
            min_value=1,
            max_value=365,  # Max one year
            value=7,        # Default to 7 days
            help="Enter an integer between 1 and 365."
        )

        # Input for BigQuery table
        bq_source_table_input = st.text_input(
            "Enter BigQuery source table (project.dataset.table):",
            placeholder="e.g., myproject.mydataset.restaurants"
        )

        # Button to trigger fetching
        if st.button("Fetch Recent Restaurants"):
            # Get inputs
            n_days = n_days_input
            bq_table_full_path = bq_source_table_input.strip()

            if not bq_table_full_path:
                st.error("BigQuery source table path is required.")
            else:
                try:
                    project_id, dataset_id, table_id = bq_table_full_path.split('.')
                    if not project_id or not dataset_id or not table_id:
                        raise ValueError("Each part of 'project.dataset.table' must be non-empty.")

                    st.info(f"Fetching restaurants from the last {n_days} days from table {bq_table_full_path}...")

                    # Call bq_utils function
                    fetched_df = get_recent_restaurants(
                        N_DAYS=n_days,
                        project_id=project_id,
                        dataset_id=dataset_id,
                        table_id=table_id
                    )

                    if fetched_df is not None and not fetched_df.empty:
                        st.session_state.recent_restaurants_df = fetched_df
                        st.success(f"Successfully fetched {len(fetched_df)} recent restaurants.")

                        # Call create_recent_restaurants_temp_table here
                        st.info("Attempting to create/update the 'recent_restaurants_temp' table in BigQuery...")
                        create_recent_restaurants_temp_table(
                            restaurants_df=fetched_df,
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                        # The create_recent_restaurants_temp_table function has its own st.success/st.error messages.

                        # Store project_id and dataset_id in session state
                        st.session_state.current_project_id = project_id
                        st.session_state.current_dataset_id = dataset_id
                        st.info(f"Project ID ({project_id}) and Dataset ID ({dataset_id}) stored in session state.")

                    elif fetched_df is not None and fetched_df.empty:
                        st.session_state.recent_restaurants_df = pd.DataFrame() # Store empty df
                        st.warning("No recent restaurants found for the given criteria.")
                    else: # Should ideally not happen if get_recent_restaurants returns empty df on error/no data
                        st.session_state.recent_restaurants_df = None
                        st.error("Failed to fetch recent restaurants. The function returned None.")

                except ValueError as ve:
                    st.error(f"Invalid BigQuery Table Path format: '{bq_table_full_path}'. Expected 'project.dataset.table'. Error: {ve}")
                    st.session_state.recent_restaurants_df = None
                except Exception as e:
                    st.error(f"An error occurred while fetching recent restaurants: {e}")
                    st.session_state.recent_restaurants_df = None

        if st.session_state.recent_restaurants_df is not None and not st.session_state.recent_restaurants_df.empty:
            st.subheader("Fetched Recent Restaurants")
            st.dataframe(st.session_state.recent_restaurants_df)

            if 'fhrsid' not in st.session_state.recent_restaurants_df.columns:
                st.warning("The fetched data does not contain an 'fhrsid' column, which is required for Gemini analysis.")
            else:
                if st.button("Run Gemini Analysis on Recent Restaurants"):
                    df_to_analyze = st.session_state.recent_restaurants_df.copy()

                    # Filter for restaurants that need analysis
                    # If 'gemini_insights' column exists, filter rows where it's null/empty
                    if 'gemini_insights' in df_to_analyze.columns:
                        df_needing_analysis = df_to_analyze[df_to_analyze['gemini_insights'].isnull() | (df_to_analyze['gemini_insights'] == '')]
                    else:
                        # If column doesn't exist, all are considered needing analysis
                        df_needing_analysis = df_to_analyze

                    if df_needing_analysis.empty:
                        st.info("No restaurants needing Gemini analysis (either all already have insights or no restaurants fetched).")
                    else:
                        st.info(f"Found {len(df_needing_analysis)} restaurants for Gemini analysis.")

                        project_id_for_gemini = st.session_state.get('current_project_id')
                        dataset_id_for_gemini = st.session_state.get('current_dataset_id')

                        if not project_id_for_gemini or not dataset_id_for_gemini:
                            st.error("Project ID or Dataset ID is not set. Please fetch recent restaurants first.")
                            return # Or st.stop()

                        fhrs_ids_list = df_needing_analysis['fhrsid'].astype(str).tolist() # Ensure FHRSIDs are strings

                        # Define Gemini Prompt (can be made configurable later)
                        gemini_prompt = (
                            "Be succint and tell me what cuisine and dishes this specific London restaurant serve. "
                            "Do not infer from the name of the restaurant, and base your answer on what you find in your search. \n" # Ensure newline is correctly escaped for a string literal
                            "Here is the Restaurant information: "
                        )

                        try:
                            # Call the Gemini analysis function
                            st.info(f"Calling Gemini for {len(fhrs_ids_list)} FHRSIDs. Project: {project_id_for_gemini}, Dataset: {dataset_id_for_gemini}")
                            gemini_results_df = call_gemini_with_fhrs_data(
                                project_id=project_id_for_gemini,
                                dataset_id=dataset_id_for_gemini,
                                gemini_prompt=gemini_prompt,
                                fhrs_ids=fhrs_ids_list
                            )

                            if gemini_results_df is not None and not gemini_results_df.empty:
                                st.success(f"Gemini analysis completed for {len(gemini_results_df)} restaurants.")

                                # Merge results back into the main session state DataFrame
                                # Ensure 'fhrsid' in gemini_results_df is of the same type as in st.session_state.recent_restaurants_df
                                # call_gemini_with_fhrs_data returns fhrsid as it was passed (string), so it should be fine.

                                if 'gemini_insights' not in st.session_state.recent_restaurants_df.columns:
                                    st.session_state.recent_restaurants_df['gemini_insights'] = pd.NA

                                # Prepare for merge: gemini_results_df has 'fhrsid' and 'gemini_insights'
                                # Set 'fhrsid' as index for easier update/merge
                                gemini_results_df = gemini_results_df.set_index('fhrsid')

                                # Update 'gemini_insights' in the main DataFrame
                                for fhrsid, row in gemini_results_df.iterrows():
                                    insights = row['gemini_insights']
                                    # Ensure fhrsid in session state df is also string for matching
                                    st.session_state.recent_restaurants_df['fhrsid'] = st.session_state.recent_restaurants_df['fhrsid'].astype(str)
                                    st.session_state.recent_restaurants_df.loc[st.session_state.recent_restaurants_df['fhrsid'] == fhrsid, 'gemini_insights'] = insights

                                # Also set 'manual_review' to 'rejected' for these updated rows
                                if 'manual_review' not in st.session_state.recent_restaurants_df.columns:
                                    st.session_state.recent_restaurants_df['manual_review'] = pd.NA # Or appropriate default like ''

                                # FHRSIDs that received new insights
                                updated_fhrsids = gemini_results_df.index.tolist()

                                # Ensure fhrsid column in session state df is string for matching
                                st.session_state.recent_restaurants_df['fhrsid'] = st.session_state.recent_restaurants_df['fhrsid'].astype(str)

                                mask_updated = st.session_state.recent_restaurants_df['fhrsid'].isin(updated_fhrsids)
                                st.session_state.recent_restaurants_df.loc[mask_updated, 'manual_review'] = "rejected"

                                st.info("Default 'manual_review' status set to 'rejected' for analyzed restaurants.")
                                # Re-display to show this change as well
                                st.dataframe(st.session_state.recent_restaurants_df)

                            elif gemini_results_df is not None and gemini_results_df.empty:
                                st.warning("Gemini analysis ran but returned no insights.")
                            else:
                                st.error("Gemini analysis failed or returned unexpected data (None).")

                        except Exception as e:
                            st.error(f"An error occurred during Gemini analysis: {e}")

            # Add the Save button
            st.markdown("---") # Add a separator
            if st.button("Save Gemini Insights and Review Status to BigQuery"):
                # Placeholder for BQ update logic
                st.info("Attempting to save changes to BigQuery...")

                # Logic to call BQ update function will be added here in a later step.
                # For now, just identify which rows to update.
                # df_to_save = st.session_state.recent_restaurants_df[
                #     st.session_state.recent_restaurants_df['fhrsid'].isin(updated_fhrsids) # Assuming updated_fhrsids is accessible
                # ] # This assumes updated_fhrsids from the Gemini run is available in this scope.
                  # This might need adjustment: we need to save rows that have 'gemini_insights'
                  # and 'manual_review' set by this session's analysis.
                  # A better way might be to filter st.session_state.recent_restaurants_df for rows
                  # where 'gemini_insights' is not null AND 'manual_review' is 'rejected' (or similar marker).

                # Revised logic for identifying rows to save:
                # We need rows that have 'gemini_insights' populated in this session
                # and potentially 'manual_review' set to 'rejected'.
                # Let's assume any row in st.session_state.recent_restaurants_df that has a non-null 'gemini_insights'
                # and a 'manual_review' column present is a candidate for saving.

                if 'gemini_insights' in st.session_state.recent_restaurants_df.columns and \
                   'manual_review' in st.session_state.recent_restaurants_df.columns:

                    # Select rows where gemini_insights is not null (or not NA)
                    # These are the rows modified by the Gemini analysis in the current session
                    rows_with_new_insights = st.session_state.recent_restaurants_df[
                        st.session_state.recent_restaurants_df['gemini_insights'].notna() &
                        (st.session_state.recent_restaurants_df['gemini_insights'] != '')
                    ]

                    # Further, we are interested in those that also have manual_review = "rejected" (set by our script)
                    # Or more broadly, any row that has a gemini_insight now.
                    # The key columns to update are 'fhrsid', 'gemini_insights', 'manual_review'.

                    df_to_save_final = rows_with_new_insights[['fhrsid', 'gemini_insights', 'manual_review']].copy()

                    if not df_to_save_final.empty:
                        st.write("Data to be saved to BigQuery:")
                        st.dataframe(df_to_save_final)
                        # Actual call to bq_utils function will be here:
                        # e.g., success = update_restaurant_analysis_data(df_to_save_final, project_id, dataset_id, table_id)
                        # if success:
                        #   st.success("Successfully saved changes to BigQuery.")
                        # else:
                        #   st.error("Failed to save changes to BigQuery.")
                        # st.warning("BigQuery update functionality is not yet fully implemented.") # Remove this line

                        # Get project_id, dataset_id, table_id from the input field
                        # This assumes bq_source_table_input is still available and holds the correct table path
                        bq_full_path_save = bq_source_table_input.strip() # bq_source_table_input is from the top of the section

                        if not bq_full_path_save:
                            st.error("BigQuery source table path is missing. Cannot save.")
                        else:
                            try:
                                project_id_save, dataset_id_save, table_id_save = bq_full_path_save.split('.')
                                if not project_id_save or not dataset_id_save or not table_id_save:
                                    raise ValueError("Each part of 'project.dataset.table' must be non-empty for saving.")

                                st.info(f"Saving updates for {len(df_to_save_final)} restaurants to {bq_full_path_save}...")

                                success = update_gemini_and_review_in_bq(
                                    project_id=project_id_save,
                                    dataset_id=dataset_id_save,
                                    table_id=table_id_save,
                                    df_updates=df_to_save_final
                                )

                                if success:
                                    # User message is handled by the bq_utils function, but an additional one here is fine.
                                    st.success("Successfully saved changes to BigQuery.")
                                    # Optionally, clear or mark saved rows if needed, or prompt user to re-fetch.
                                    # For now, the data remains in st.session_state.recent_restaurants_df as is.
                                    # If user runs Gemini again, it won't re-analyze these due to existing insights.
                                else:
                                    # Error message is handled by bq_utils, but one here is fine.
                                    st.error("Failed to save changes to BigQuery. Check logs for details.")

                            except ValueError as ve:
                                st.error(f"Invalid BigQuery Table Path for saving: '{bq_full_path_save}'. Error: {ve}")
                            except Exception as e:
                                st.error(f"An unexpected error occurred during the save operation: {e}")
                    else:
                        st.warning("No new Gemini insights or 'rejected' reviews to save.")
                else:
                    st.warning("Required columns ('gemini_insights', 'manual_review') not found in the DataFrame. Cannot save.")

        elif st.session_state.recent_restaurants_df is not None and st.session_state.recent_restaurants_df.empty:
            st.info("No recent restaurants to display based on the last fetch attempt.")


if __name__ == "__main__":
    main_ui()
