import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import datetime

from app.services.bq_utils import (
    load_filtered_data_from_bq,
    execute_gemini_enrichment,
    bulk_update_reviews,
    get_distinct_local_authorities,
    upsert_agent_insight
)
from app.core.data_processing import (
    load_data_from_csv,
    parse_bq_path
)
from app.services.agent_orchestrator import get_agent_insight

def display_data(data_to_display: List[Dict[str, Any]]):
    if not data_to_display:
        st.info("No data to display.")
        return None
    df = pd.DataFrame(data_to_display)
    # Using on_select to enable row selection
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        key="review_queue_table"
    )
    return event

def get_selected_rows(selection_event: Any, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts selected rows from the Streamlit dataframe selection event.
    Handles different event structures for compatibility.
    """
    selected_indices = []
    if selection_event:
        # Streamlit 1.35+ returns {'selection': {'rows': [0, 1], 'columns': []}}
        if isinstance(selection_event, dict) and "selection" in selection_event:
             selected_indices = selection_event["selection"].get("rows", [])
        else:
             # Fallback for direct structure or older versions/other widgets
             # Try dictionary access first
             try:
                 selected_indices = selection_event.get("rows", [])
             except AttributeError:
                 # Fallback if it's an object with attributes
                 selected_indices = getattr(selection_event, "rows", [])

    selected_rows = []
    if selected_indices:
        df = pd.DataFrame(data)
        # Ensure indices are valid and convert to python ints if needed
        valid_indices = [int(i) for i in selected_indices if i < len(df)]
        if valid_indices:
            selected_rows = df.iloc[valid_indices].to_dict('records')
            
    return selected_rows

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
        if bq_path:
            # Using session state to avoid re-fetching on every interaction
            if 'la_options' not in st.session_state:
                 with st.spinner("Loading Local Authorities..."):
                    st.session_state.la_options = get_distinct_local_authorities(project_id, dataset_id, table_id)
            
            if st.button("Refresh Authorities"):
                st.session_state.pop('la_options', None)
                with st.spinner("Refreshing Local Authorities..."):
                    st.session_state.la_options = get_distinct_local_authorities(project_id, dataset_id, table_id)
                    st.success(f"Refreshed list. Found {len(st.session_state.la_options)} authorities.")
            
            la_options = st.session_state.la_options if 'la_options' in st.session_state else []
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
        
        selection_event = display_data(st.session_state.review_data)
        selected_rows = get_selected_rows(selection_event, st.session_state.review_data)
        
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Analysis & Insights")
            
            tab_gemini, tab_agent = st.tabs(["Gemini Analysis (Batch)", "Agent Research (Selected)"])
            
            with tab_gemini:
                st.write("Run Gemini analysis on ALL filtered data.")
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
            
            with tab_agent:
                st.write("Generate deep insights for SELECTED restaurants using the Agent.")
                if not selected_rows:
                    st.info("Select rows in the table above to enable Agent Research.")
                    st.button("Generate Agent Insights", disabled=True)
                else:
                    st.write(f"Selected {len(selected_rows)} restaurants.")
                    if st.button("Generate Agent Insights"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        success_count = 0
                        total = len(selected_rows)
                        
                        for i, restaurant in enumerate(selected_rows):
                            business_name = restaurant.get('businessname', 'Unknown')
                            status_text.text(f"Processing {i+1}/{total}: {business_name}")
                            
                            insight = get_agent_insight(restaurant)
                            
                            if insight:
                                if 'updated_at' not in insight:
                                    insight['updated_at'] = datetime.datetime.now().isoformat()
                                
                                upsert_success = upsert_agent_insight(project_id, dataset_id, "restaurant_agent_insights", insight)
                                if upsert_success:
                                    success_count += 1
                                else:
                                    st.error(f"Failed to save insight for {business_name}")
                            else:
                                st.error(f"Agent failed for {business_name}")
                                
                            progress_bar.progress((i + 1) / total)
                            
                        status_text.text(f"Completed! Successfully processed {success_count}/{total}.")
                        if success_count == total:
                            st.success("All insights generated and saved.")
                        else:
                            st.warning("Some insights failed.")

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
