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
    update_rows_in_bigquery, # Added for Update Fields
    sanitize_column_name,
    write_to_bigquery,
    load_all_data_from_bq, # Added import
    append_to_bigquery, # Added for new flow
    execute_merge_query, # Added for MERGE operation
    BigQueryExecutionError,  # Added import
    DataFrameConversionError, # Added import
    execute_gemini_enrichment, # Added for Gemini Analysis
    load_filtered_data_from_bq, # Added for Export Section
    bulk_update_reviews # Added for Bulk Update Section
)
from data_processing import load_json_from_local_file_path, load_master_data, process_and_update_master_data
from data_processing import load_data_from_csv # Added for Update Fields
from auth.firebase_auth import AuthManager
from login import login_page

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
    Fetches API data for all valid coordinates, handling pagination, and aggregates the results.
    """
    all_api_establishments = []
    st.info(f"Found {len(valid_coords)} valid coordinate pairs. Fetching data for each...")
    for lon, lat in valid_coords:
        page = 1
        while True:
            st.write(f"Fetching data for Longitude: {lon}, Latitude: {lat}, Page: {page}...")
            api_response = fetch_api_data(lon, lat, max_results, page)
            time.sleep(4)  # Respect API rate limits

            if api_response:
                establishments_list = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
                if establishments_list is None:  # Handle cases where EstablishmentDetail might be null
                    establishments_list = []

                num_results_this_call = len(establishments_list)
                st.info(f"API call for ({lon}, {lat}, Page: {page}) returned {num_results_this_call} establishments.")
                all_api_establishments.extend(establishments_list)

                if num_results_this_call < max_results:
                    break  # Exit loop if the results are less than max_results, indicating the last page
                else:
                    page += 1  # Otherwise, increment page and continue
            else:
                st.error(f"Failed to fetch data for Longitude: {lon}, Latitude: {lat}, Page: {page}.")
                break  # Exit loop on API error
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
        # RatingDate removed as it's not in ORIGINAL_COLUMNS_TO_KEEP and causes a warning
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


def display_new_restaurants(new_restaurants: List[Dict[str, Any]]):
    """
    Displays the list of newly discovered restaurants using a Streamlit dataframe
    with a configured LinkColumn for the Google Maps URL.
    """
    if not new_restaurants:
        st.info("No new restaurants to display.")
        return

    st.subheader(f"Newly identified restaurants ({len(new_restaurants)})")
    
    df = pd.DataFrame(new_restaurants)
    
    # Configure the 'Maps Link' column to render as a clickable link
    st.dataframe(
        df,
        column_config={
            "Maps Link": st.column_config.LinkColumn(
                "Research on Maps",
                display_text="Search Maps"
            )
        },
        hide_index=True,
    )

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

    # Store newly identified restaurants in session state for later review
    if new_restaurants:
        st.session_state.new_restaurants_to_review = new_restaurants
        st.subheader(f"Newly identified restaurants from this fetch ({len(new_restaurants)}). Please review below:")
    else:
        st.session_state.new_restaurants_to_review = []
        st.info("No new restaurants identified in this fetch.")

    # 7. Display data
    # This displays the initial master_restaurant_data. If the display should reflect the appended data,
    # it would need to either re-load from BQ or combine master_restaurant_data + new_restaurants (if df versions are compatible).
    # For now, sticking to displaying initial load.
    st.info("Displaying master data loaded from BigQuery (before current API fetch append).") # Modified the info message slightly
    display_data(master_restaurant_data) # Explicitly display the loaded master data

    # The function is expected to return List[Dict[str, Any]].
    # Returning the initial master_data for now, as the old _write_data_to_bigquery
    # was also based on the (then modified) master_data.
    # If the expectation is to return the "complete" data after append, this would need adjustment.
    return master_restaurant_data

def update_restaurant_review_status(fhrsid: str, new_status: str, bq_full_path_str: str):
    """
    Updates the manual_review status for a given FHRSID in BigQuery and
    removes the restaurant from the session state list if successful.
    """
    if not bq_full_path_str:
        st.error("BigQuery Table Path is missing for update operation.")
        return

    try:
        project_id, dataset_id, table_id = bq_full_path_str.split('.')
        if not all([project_id, dataset_id, table_id]):
            st.error(f"Invalid BigQuery Table Path format: '{bq_full_path_str}'. Expected 'project.dataset.table' with non-empty parts.")
            return
    except ValueError:
        st.error(f"Invalid BigQuery Table Path format: '{bq_full_path_str}'. Expected 'project.dataset.table'.")
        return

    update_data = {"manual_review": new_status}
    success = update_rows_in_bigquery(project_id, dataset_id, table_id, fhrsid, update_data)

    if success:
        st.success(f"Successfully updated FHRSID {fhrsid} to '{new_status}' in BigQuery.")
        # Remove the reviewed item from the session state list
        st.session_state.new_restaurants_to_review = [
            r for r in st.session_state.new_restaurants_to_review if r.get('FHRSID') != fhrsid
        ]
        st.rerun() # Rerun to update the display of new restaurants
    else:
        st.error(f"Failed to update FHRSID {fhrsid} to '{new_status}' in BigQuery.")


def get_iap_user_email() -> str | None:
    """
    Attempts to retrieve the user's email from the IAP header.
    Note: This uses an internal Streamlit API and might be fragile.
    """
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        email = headers.get("X-Goog-Authenticated-User-Email")
        if email:
            # The email often comes as "accounts.google.com:user@example.com"
            # We want to extract just "user@example.com"
            if email.startswith("accounts.google.com:"):
                return email.split(":", 1)[1]
            return email
    except Exception:
        # Handle cases where the internal API changes or is not available
        pass
    return None

def main_ui():
    auth_manager = AuthManager()
    
    # 1. Check for auth token in query params or cookies
    auth_manager.check_auth()
    
    # 2. Force login if not authenticated
    if not auth_manager.is_authenticated():
        login_page(auth_manager)
        return

    st.title("Food Standards Agency API Explorer")

    # Display authenticated user
    user_email = auth_manager.get_user_email()
    if user_email:
        st.sidebar.success(f"Logged in as: {user_email}")
        if st.sidebar.button("Sign Out"):
            auth_manager.sign_out()
            st.rerun()
    else:
        # Fallback to IAP if configured but not logged in via Firebase
        iap_email = get_iap_user_email()
        if iap_email:
            st.sidebar.success(f"Logged in via IAP as: {iap_email}")
        else:
            st.sidebar.warning("Not logged in.")

    # Initialize session state variables if they don't exist
    if 'recent_restaurants_df' not in st.session_state:
        st.session_state.recent_restaurants_df = None
    if 'current_project_id' not in st.session_state:
        st.session_state['current_project_id'] = None
    if 'current_dataset_id' not in st.session_state:
        st.session_state['current_dataset_id'] = None
    if 'displaying_genai_temp' not in st.session_state: # Initialize new session state variable
        st.session_state.displaying_genai_temp = False
    if 'new_restaurants_to_review' not in st.session_state: # Initialize new session state variable for new restaurants
        st.session_state.new_restaurants_to_review = []


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

    # Display newly discovered restaurants if available in session state
    if st.session_state.get('new_restaurants_to_review'):
        display_new_restaurants(st.session_state.new_restaurants_to_review)

    st.divider()
    st.subheader("Gemini Intelligence Analysis")
    st.markdown("Run advanced analysis on recently added restaurants using Gemini.")
    
    col1, col2 = st.columns(2)
    with col1:
        connection_id_input = st.text_input("BigQuery Connection ID", value="eu.gemini", help="Region.ConnectionName (e.g. eu.gemini)")
    with col2:
        days_recent_input = st.number_input("Days Lookback", min_value=1, value=33, help="Number of days to consider a restaurant 'recent'.")

    if st.button("Run Gemini Analysis"):
        if not bq_full_path_ui:
             st.error("Please enter the BigQuery Table Path above (in the Fetch Data section) to identify the master table.")
        else:
            try:
                project_id, dataset_id, table_id = bq_full_path_ui.split('.')
                if not all([project_id, dataset_id, table_id]):
                     st.error("Invalid BigQuery Path format.")
                else:
                    with st.spinner("Running Gemini Analysis... Check terminal for detailed logs."):
                        success = execute_gemini_enrichment(
                            project_id=project_id,
                            dataset_id=dataset_id,
                            master_table_id=table_id,
                            connection_id=connection_id_input,
                            model_endpoint="gemini-3-pro",
                            days_recent=days_recent_input
                        )
                        if success:
                            st.success("Gemini Analysis completed successfully! Insights have been merged into the master table.")
                        else:
                            st.error("Gemini Analysis failed. Check the logs for details.")
            except ValueError:
                st.error("Invalid BigQuery Path format. Expected 'project.dataset.table'.")

    st.divider()
    st.subheader("Export Filtered Data")
    st.markdown("Query and export specific segments of the master data.")

    # Defaults from SCRIPT1
    default_excluded_locs = [
        "Westminster", "City of London Corporation", "Tower Hamlets", 
        "Kingston-Upon-Thames", "Camden", "Kensington and Chelsea", 
        "Hackney", "Islington", "Hammersmith and Fulham"
    ]
    default_review_statuses = ["pending", "not reviewed"]

    with st.form("export_form"):
        c1, c2 = st.columns(2)
        with c1:
            export_days_input = st.number_input("Filter by 'First Seen' within last X days (0 to disable)", value=33, min_value=0)
        with c2:
            export_status_input = st.multiselect(
                "Filter by Manual Review Status", 
                options=["pending", "not reviewed", "accepted", "rejected", "unsure"],
                default=default_review_statuses
            )
        
        export_excluded_locs_input = st.multiselect(
            "Exclude Local Authorities (Default list from analysis script)",
            options=default_excluded_locs + ["Other..."], # Add dummy "Other" if we want to expand later, but for now just the known list is safer
            default=default_excluded_locs
        )
        
        # Allow user to add custom exclusions via text if needed? 
        # For now, keep it simple as requested "same filters as in script1"
        
        submitted = st.form_submit_button("Run Query & Preview")

    if submitted:
        if not bq_full_path_ui:
            st.error("Please enter the BigQuery Table Path in the 'Fetch Data' section above.")
        else:
            try:
                project_id, dataset_id, table_id = bq_full_path_ui.split('.')
                if not all([project_id, dataset_id, table_id]):
                     st.error("Invalid BigQuery Path format.")
                else:
                    with st.spinner("Running parameterized query..."):
                        # Handle 0 days as None (disable filter)
                        days_param = export_days_input if export_days_input > 0 else None
                        
                        results = load_filtered_data_from_bq(
                            project_id=project_id,
                            dataset_id=dataset_id,
                            table_id=table_id,
                            days_filter=days_param,
                            review_status_filter=export_status_input,
                            excluded_locations=export_excluded_locs_input
                        )
                        
                        if results:
                            st.success(f"Found {len(results)} records.")
                            df_results = pd.DataFrame(results)
                            st.dataframe(df_results)
                            
                            # CSV Download
                            csv = df_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name='fsa_export_filtered.csv',
                                mime='text/csv',
                            )
                        else:
                            st.warning("No records found matching these criteria.")
                            
            except ValueError:
                st.error("Invalid BigQuery Path format. Expected 'project.dataset.table'.")

    st.divider()
    st.subheader("Bulk Update Manual Reviews")
    st.markdown("Upload a CSV to bulk update the 'manual_review' status for specific restaurants.")
    
    uploaded_file = st.file_uploader("Upload CSV (Required columns: 'fhrsid', 'manual_review')", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_updates = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['fhrsid', 'manual_review']
            if not all(col in df_updates.columns for col in required_cols):
                st.error(f"CSV is missing required columns. Found: {df_updates.columns.tolist()}. Expected: {required_cols}")
            else:
                st.info("Preview of updates:")
                st.dataframe(df_updates.head())
                st.write(f"Total rows to update: {len(df_updates)}")
                
                if st.button("Execute Bulk Update"):
                    if not bq_full_path_ui:
                        st.error("Please enter the BigQuery Table Path in the 'Fetch Data' section above.")
                    else:
                        try:
                            project_id, dataset_id, table_id = bq_full_path_ui.split('.')
                            if not all([project_id, dataset_id, table_id]):
                                st.error("Invalid BigQuery Path format.")
                            else:
                                with st.spinner("Executing bulk update (uploading temp table -> merging -> cleaning up)..."):
                                    # Ensure fhrsid is string for consistency
                                    df_updates['fhrsid'] = df_updates['fhrsid'].astype(str)
                                    
                                    success = bulk_update_reviews(
                                        project_id=project_id,
                                        dataset_id=dataset_id,
                                        target_table_id=table_id,
                                        df_updates=df_updates
                                    )
                                    
                                    if success:
                                        st.success("Bulk update completed successfully!")
                                    else:
                                        st.error("Bulk update failed. Check console logs for details.")
                        except ValueError:
                            st.error("Invalid BigQuery Path format. Expected 'project.dataset.table'.")
                            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")


if __name__ == "__main__":
    main_ui()
