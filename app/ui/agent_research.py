import streamlit as st
import pandas as pd
import datetime
from app.services.agent_orchestrator import get_agent_insight
from app.services.bq_utils import upsert_agent_insight, load_specific_agent_insights

def handle_insight_generation(targets, project_id, dataset_id, progress_bar, status_text):
    """
    Orchestrates the agent insight generation for a list of targets.
    Updates session state with processed IDs.
    """
    success_count = 0
    total = len(targets)
    processed_fhrsids = []
    
    # Reset session state for new batch
    st.session_state['latest_batch_fhrsids'] = []
    st.session_state['show_latest_insights'] = False

    for i, restaurant in enumerate(targets):
        business_name = restaurant.get('businessname', 'Unknown')
        status_text.text(f"Processing {i+1}/{total}: {business_name}")
        
        # Call Agent
        insight = get_agent_insight(restaurant)
        
        if insight:
            if 'updated_at' not in insight:
                insight['updated_at'] = datetime.datetime.now().isoformat()
            
            upsert_success = upsert_agent_insight(project_id, dataset_id, "restaurant_agent_insights", insight)
            if upsert_success:
                success_count += 1
                if 'fhrsid' in restaurant:
                    processed_fhrsids.append(str(restaurant['fhrsid']))
            else:
                st.error(f"Failed to save insight for {business_name}")
        else:
            st.error(f"Agent failed for {business_name}")
            
        progress_bar.progress((i + 1) / total)
    
    # Update Session State
    st.session_state['latest_batch_fhrsids'] = processed_fhrsids
    if processed_fhrsids:
        st.session_state['show_latest_insights'] = True
    
    return success_count, total

def render_agent_research_tab(project_id, dataset_id, selected_rows):
    st.write("Generate deep insights for restaurants using the Agent.")
    
    targets = []
    
    # Selection Logic: Always use Selected Rows
    if not selected_rows:
        st.info("Select rows in the table above to enable Agent Research.")
    else:
        targets = selected_rows
        st.write(f"Targeting {len(targets)} selected restaurants.")

    # Action Button
    if targets:
        if st.button(f"Generate Agent Insights ({len(targets)} Restaurants)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count, total = handle_insight_generation(
                targets, project_id, dataset_id, progress_bar, status_text
            )
                
            status_text.text(f"Completed! Successfully processed {success_count}/{total}.")
            if success_count == total:
                st.success("All insights generated and saved.")
            else:
                st.warning("Some insights failed.")
    else:
        st.button("Generate Agent Insights", disabled=True, help="Please select rows in the table above to enable this button.")

    # New Results Section
    if st.session_state.get('show_latest_insights') and st.session_state.get('latest_batch_fhrsids'):
        st.divider()
        st.subheader("New Agent Insights")
        with st.expander("View Latest Batch Results", expanded=True):
            with st.spinner("Fetching latest insights..."):
                latest_data = load_specific_agent_insights(
                    project_id, dataset_id, st.session_state['latest_batch_fhrsids']
                )
                if latest_data:
                    st.dataframe(pd.DataFrame(latest_data))
                else:
                    st.info("No insights found in BigQuery for this batch yet.")
