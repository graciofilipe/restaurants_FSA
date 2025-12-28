# Standard Library
import json
import time
import os
import mimetypes
from datetime import datetime
from typing import List, Dict, Any

# Ensure .html is served as text/html even if system mime.types is missing
mimetypes.add_type('text/html', '.html')
mimetypes.add_type('application/javascript', '.js')

# Third-party
import pandas as pd
import streamlit as st
from google.cloud import bigquery
import streamlit.components.v1 as components

# Local Modules
from api_client import fetch_api_data
from bq_utils import (
    update_rows_in_bigquery,
    sanitize_column_name,
    write_to_bigquery,
    load_all_data_from_bq,
    append_to_bigquery,
    execute_merge_query,
    BigQueryExecutionError,
    DataFrameConversionError,
    execute_gemini_enrichment,
    load_filtered_data_from_bq,
    bulk_update_reviews
)
from data_processing import load_json_from_local_file_path, load_master_data, process_and_update_master_data
from data_processing import load_data_from_csv

def display_data(data_to_display: List[Dict[str, Any]]):
    """
    Displays the given data using Streamlit, primarily as a Pandas DataFrame.
    """
    try:
        if not data_to_display:
            st.warning("No restaurant data to display.")
            return

        valid_items_for_df = [item for item in data_to_display if isinstance(item, dict)]
        
        if valid_items_for_df:
            df = pd.json_normalize(valid_items_for_df)
            st.dataframe(df)
        
    except Exception as e: 
        st.error(f"Error displaying DataFrame: {e}")

# Helper functions for handle_fetch_data_action
def _parse_coordinates(coordinate_pairs_str: str) -> List[tuple[float, float]]:
    valid_coords = []
    coordinate_lines = coordinate_pairs_str.strip().split('\n')
    for i, line in enumerate(coordinate_lines):
        line = line.strip()
        if not line: continue
        try:
            lon_str, lat_str = line.split(',')
            valid_coords.append((float(lon_str.strip()), float(lat_str.strip())))
        except ValueError:
            st.error(f"Error parsing coordinate line {i+1}: '{line}'.")
    return valid_coords

def _fetch_data_for_all_coordinates(valid_coords: List[tuple[float, float]], max_results: int) -> List[Dict[str, Any]]:
    all_api_establishments = []
    for lon, lat in valid_coords:
        page = 1
        while True:
            api_response = fetch_api_data(lon, lat, max_results, page)
            time.sleep(1) 
            if api_response:
                establishments = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
                if establishments is None: establishments = []
                all_api_establishments.extend(establishments)
                if len(establishments) < max_results: break
                page += 1
            else: break
    return all_api_establishments

def display_new_restaurants(new_restaurants: List[Dict[str, Any]]):
    if not new_restaurants: return
    st.subheader(f"Newly identified restaurants ({len(new_restaurants)})")
    df = pd.DataFrame(new_restaurants)
    st.dataframe(df, hide_index=True)

def handle_fetch_data_action(coordinate_pairs_str: str, max_results: int, bq_full_path_str: str) -> List[Dict[str, Any]]:
    valid_coords = _parse_coordinates(coordinate_pairs_str)
    if not valid_coords: 
        st.error("No valid coordinates.")
        return []
    
    try:
        project_id, dataset_id, table_id = bq_full_path_str.split('.')
    except ValueError:
        st.error("Invalid BigQuery Path.")
        return []

    all_api_establishments = _fetch_data_for_all_coordinates(valid_coords, max_results)
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}

    master_restaurant_data = load_master_data(project_id, dataset_id, table_id, load_all_data_from_bq)
    new_restaurants = process_and_update_master_data(master_restaurant_data, combined_api_data)

    if new_restaurants:
        st.session_state.new_restaurants_to_review = new_restaurants
        st.success(f"Found {len(new_restaurants)} new restaurants!")
    
    display_data(master_restaurant_data)
    return master_restaurant_data

def main_ui():
    # Initialize session state variables
    if 'app_entered' not in st.session_state:
        st.session_state.app_entered = False

    if not st.session_state.app_entered:
        st.title("FSA API Explorer")
        st.write("Welcome to the Food Standards Agency API Explorer. Explore and analyze food hygiene ratings.")
        if st.button("Enter App"):
            st.session_state.app_entered = True
            st.rerun()
        return

    st.title("Food Standards Agency API Explorer")

    # Initialize session state variables
    if 'new_restaurants_to_review' not in st.session_state:
        st.session_state.new_restaurants_to_review = []

    st.subheader("Fetch API Data and Update Master List")
    coordinate_pairs_input = st.text_area("Enter longitude,latitude pairs (one per line):")
    max_results_input_ui = st.number_input("Enter Max Results", min_value=1, max_value=5000, value=200)
    bq_full_path_ui = st.text_input("Enter BigQuery Table Path (project.dataset.table)")

    if st.button("Fetch Data"):
        handle_fetch_data_action(coordinate_pairs_input, max_results_input_ui, bq_full_path_ui)

    if st.session_state.get('new_restaurants_to_review'):
        display_new_restaurants(st.session_state.new_restaurants_to_review)

    st.divider()
    st.subheader("Gemini Intelligence Analysis")
    col1, col2 = st.columns(2)
    with col1:
        connection_id_input = st.text_input("BigQuery Connection ID", value="eu.gemini")
    with col2:
        days_recent_input = st.number_input("Days Lookback", min_value=1, value=33)

    if st.button("Run Gemini Analysis"):
        if bq_full_path_ui:
            try:
                p, d, t = bq_full_path_ui.split('.')
                with st.spinner("Analyzing..."):
                    if execute_gemini_enrichment(p, d, t, connection_id_input, days_recent=days_recent_input):
                        st.success("Analysis Complete!")
            except ValueError: st.error("Invalid path.")

    st.divider()
    st.subheader("Export Filtered Data")
    with st.form("export_form"):
        c1, c2 = st.columns(2)
        with c1:
            export_days_input = st.number_input("Filter by 'First Seen' (days)", value=33, min_value=0)
        with c2:
            export_status_input = st.multiselect("Review Status", options=["pending", "not reviewed", "accepted", "rejected"], default=["pending", "not reviewed"])
        
        submitted = st.form_submit_button("Run Query & Preview")

    if submitted:
        if bq_full_path_ui:
            try:
                p, d, t = bq_full_path_ui.split('.')
                results = load_filtered_data_from_bq(p, d, t, days_filter=export_days_input, review_status_filter=export_status_input)
                if results:
                    st.dataframe(pd.DataFrame(results))
            except ValueError: st.error("Invalid path.")

    st.divider()
    st.subheader("Bulk Update Manual Reviews")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df_updates = pd.read_csv(uploaded_file)
        print(f"DEBUG: Uploaded CSV columns: {df_updates.columns.tolist()}")
        st.dataframe(df_updates.head())
        if st.button("Execute Bulk Update"):
            if bq_full_path_ui:
                try:
                    p, d, t = bq_full_path_ui.split('.')
                    affected_rows = bulk_update_reviews(p, d, t, df_updates)
                    if affected_rows is not None:
                        st.success(f"Bulk update successful! {affected_rows} rows updated.")
                    else:
                        st.error("Bulk update failed. Check logs for details.")
                except ValueError: st.error("Invalid path.")

if __name__ == "__main__":
    main_ui()