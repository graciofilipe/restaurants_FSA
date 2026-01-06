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
from app.services.api_client import fetch_api_data
from app.services.bq_utils import (
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
from app.core.data_processing import (
    load_json_from_local_file_path,
    load_data_from_csv,
    parse_coordinates,
    run_data_synchronization,
    parse_bq_path
)

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

def display_new_restaurants(new_restaurants: List[Dict[str, Any]]):
    if not new_restaurants: return
    st.subheader(f"Newly identified restaurants ({len(new_restaurants)})")
    df = pd.DataFrame(new_restaurants)
    st.dataframe(df, hide_index=True)

def handle_fetch_data_action(coordinate_pairs_str: str, max_results: int, bq_full_path_str: str) -> List[Dict[str, Any]]:
    valid_coords, errors = parse_coordinates(coordinate_pairs_str)
    for error in errors:
        st.error(error)

    if not valid_coords: 
        st.error("No valid coordinates.")
        return []
    
    try:
        project_id, dataset_id, table_id = parse_bq_path(bq_full_path_str)
    except ValueError as e:
        st.error(str(e))
        return []

    with st.spinner("Synchronizing data (API -> Processing -> Master Data)..."):
        try:
            master_restaurant_data, new_restaurants, summary_msg = run_data_synchronization(
                valid_coords, max_results, project_id, dataset_id, table_id
            )
            
            if master_restaurant_data:
                st.success(f"Loaded {len(master_restaurant_data)} records from BigQuery.")
            else:
                st.info("Master table is empty or returned no data.")
                
            if summary_msg:
                st.info(summary_msg)

            if new_restaurants:
                st.session_state.new_restaurants_to_review = new_restaurants
                st.success(f"Found {len(new_restaurants)} new restaurants!")
            
            display_data(master_restaurant_data)
            return master_restaurant_data

        except Exception as e:
            st.error(f"Data synchronization failed: {e}")
            return []

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
                p, d, t = parse_bq_path(bq_full_path_ui)
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
                p, d, t = parse_bq_path(bq_full_path_ui)
                results = load_filtered_data_from_bq(p, d, t, days_filter=export_days_input, review_status_filter=export_status_input)
                if results:
                    st.dataframe(pd.DataFrame(results))
            except ValueError: st.error("Invalid path.")

    st.divider()
    st.subheader("Bulk Update Manual Reviews")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df_updates = load_data_from_csv(uploaded_file)
            print(f"DEBUG: Uploaded CSV columns: {df_updates.columns.tolist()}")
            st.dataframe(df_updates.head())
            
            if st.button("Execute Bulk Update"):
                if bq_full_path_ui:
                    try:
                        p, d, t = parse_bq_path(bq_full_path_ui)
                        success, message = bulk_update_reviews(p, d, t, df_updates)
                        if success:
                            st.success(f"Bulk update successful! {message}")
                        else:
                            st.error(f"Bulk update failed: {message}")
                    except ValueError: st.error("Invalid path.")
                else:
                    st.error("Please enter a BigQuery Table Path.")
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main_ui()