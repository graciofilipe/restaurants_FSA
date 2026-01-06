import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from app.services.bq_utils import (
    load_filtered_data_from_bq,
    execute_gemini_enrichment,
    bulk_update_reviews,
    get_distinct_local_authorities
)
from app.core.data_processing import (
    load_data_from_csv,
    parse_bq_path
)

def display_data(data_to_display: List[Dict[str, Any]]):
    if not data_to_display:
        st.info("No data to display.")
        return
    df = pd.DataFrame(data_to_display)
    st.dataframe(df)

def main_ui():
    st.set_page_config(layout="wide", page_title="FSA Restaurant Reviewer")
    st.title("FSA Restaurant Reviewer")

    if 'review_data' not in st.session_state:
        st.session_state.review_data = None

    with st.sidebar:
        st.header("Configuration")
        default_bq_path = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
        bq_path = st.text_input("BigQuery Master Table", value=default_bq_path)
        
        try:
            project_id, dataset_id, table_id = parse_bq_path(bq_path)
        except ValueError:
            st.error("Invalid BQ Path")
            return

        st.divider()
        st.subheader("Review Parameters")
        
        days_lookback = st.number_input("First Seen within last X days", min_value=1, value=7)
        
        status_options = ["not reviewed", "pending", "accepted", "rejected"]
        selected_statuses = st.multiselect("Review Status", options=status_options, default=["not reviewed"])
        
        # Exclude Authorities
        # Fetch dynamically if path is valid
        if bq_path:
            # Check if we should fetch (maybe adding a button or just doing it if not too slow)
            # st.spinner might flicker. Let's try to cache this or just load it.
            # Ideally use @st.cache_data for this function in a real app.
            # For now, I'll put it inside a check to verify connection/existence if possible, or just try/except inside get_distinct...
            
            # Using session state to avoid re-fetching on every interaction
            if 'la_options' not in st.session_state:
                 with st.spinner("Loading Local Authorities..."):
                    st.session_state.la_options = get_distinct_local_authorities(project_id, dataset_id, table_id)
            
            la_options = st.session_state.la_options
            excluded_las = st.multiselect("Exclude Local Authorities", options=la_options)
        else:
            excluded_las = []

        if st.button("Load Data for Review", type="primary"):
            with st.spinner("Loading filtered data..."):
                data = load_filtered_data_from_bq(
                    project_id, dataset_id, table_id,
                    days_filter=days_lookback,
                    review_status_filter=selected_statuses,
                    excluded_locations=excluded_las
                )
                st.session_state.review_data = data
                if not data:
                    st.warning("No records found matching criteria.")
                else:
                    st.success(f"Loaded {len(data)} records.")

    # Main Area
    if st.session_state.review_data:
        st.subheader(f"Review Queue ({len(st.session_state.review_data)} records)")
        display_data(st.session_state.review_data)
        
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Gemini Analysis")
            st.write("Run Gemini analysis on the filtered data matching the criteria above.")
            connection_id = st.text_input("Connection ID", value="eu.gemini")
            if st.button("Run Gemini Analysis"):
                with st.spinner("Running Gemini Analysis... this may take a while."):
                    success = execute_gemini_enrichment(
                        project_id, dataset_id, table_id,
                        connection_id=connection_id,
                        days_recent=days_lookback,
                        review_status_filter=selected_statuses,
                        excluded_locations=excluded_las
                    )
                    if success:
                        st.success("Analysis Complete! Reload data to see results.")
                    else:
                        st.error("Analysis Failed. Check logs.")

        with c2:
            st.subheader("Export Data")
            if st.session_state.review_data:
                df = pd.DataFrame(st.session_state.review_data)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='restaurant_review.csv',
                    mime='text/csv',
                )

        st.divider()
        st.subheader("Bulk Update")
        uploaded_file = st.file_uploader("Upload Reviewed CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df_updates = load_data_from_csv(uploaded_file)
                st.dataframe(df_updates.head())
                if st.button("Execute Bulk Update"):
                    with st.spinner("Updating..."):
                        success, message = bulk_update_reviews(project_id, dataset_id, table_id, df_updates)
                        if success:
                            st.success(f"Update successful: {message}")
                        else:
                            st.error(f"Update failed: {message}")
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main_ui()
