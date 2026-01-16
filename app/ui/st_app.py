import streamlit as st
import time
import pandas as pd
from app.services.bq_utils import (
    load_all_data_from_bq, 
    execute_gemini_enrichment, 
    get_distinct_local_authorities,
    get_distinct_outcodes,
    load_filtered_data_from_bq
)
from app.core.data_processing import (
    load_data_from_csv, 
    run_data_synchronization, 
    enhance_dataframe_with_insights
)

st.set_page_config(page_title="FSA Restaurant Explorer", layout="wide")

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

def display_data(df, key=None):
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        hide_index=True,
        key=key
    )
    return event

def get_selected_rows(event, df):
    if event and event.selection and event.selection.rows:
        return df.iloc[event.selection.rows]
    return None

def render_insights_details(row):
    """
    Renders the detailed 6-pillar insights for a single selected row.
    """
    # Handle case-insensitive column names (BQ often returns lowercase)
    biz_name = row.get('BusinessName') or row.get('businessname') or "Unknown Restaurant"
    st.markdown(f"### 🍽️ Profiling: {biz_name}")
    
    if not row.get('detailed_insights'):
        st.info("No detailed persona profile available for this restaurant yet.")
        if row.get('insight_summary'):
             st.write(f"**Legacy Summary:** {row['insight_summary']}")
        return

    insights = row['detailed_insights']
    
    # top level metric
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Persona Match Score", f"{insights.get('match_score', 'N/A')}/100")
        
    with col2:
        if insights.get('summary_reasoning'):
            st.info(f"**Verdict:** {insights['summary_reasoning']}")

    st.markdown("---")
    st.markdown("#### 🔬 The 6-Pillar Analysis")
    
    # Dynamic rendering of numbered criteria
    pillars = [k for k in insights.keys() if k and k[0].isdigit()]
    pillars.sort() # Ensure 1-6 order
    
    if not pillars:
        st.warning("No structured pillar data found in insights.")
        return

    for pillar_key in pillars:
        pillar_data = insights[pillar_key]
        pillar_title = pillar_key.replace('_', ' ').title()
        
        with st.expander(f"{pillar_title}", expanded=True):
            if isinstance(pillar_data, dict):
                score_val = pillar_data.get('score') or pillar_data.get('rating')
                
                c_score, c_details = st.columns([1, 4])
                
                with c_score:
                    if score_val:
                        st.metric("Score", f"{score_val}/5")
                    
                with c_details:
                    for k, v in pillar_data.items():
                        if k not in ['score', 'rating']:
                            nice_key = k.replace('_', ' ').capitalize()
                            st.write(f"**{nice_key}:** {v}")
            else:
                st.write(pillar_data)

def main():
    st.title("🍔 FSA Restaurant Explorer & Profiler")
    
    # --- Data Loading (Before Sidebar to enable dynamic filters) ---
    # In a production app, we would cache this or load minimal metadata first.
    # For now, we load all data to populate filters accurately.
    
    # 1. Get Config (Default or Session State could be used)
    bq_path = DEFAULT_BQ_PATH 
    # Check if user overrode it in sidebar (we need to peek or just render config there)
    # To keep dynamic filters working, we assume default or last known.
    # We will enforce the styling later in sidebar.
    
    project_id, dataset_id, table_id = bq_path.split('.')
    df_enriched = pd.DataFrame()
    
    try:
        raw_data = load_all_data_from_bq(project_id, dataset_id, table_id)
        df_master = pd.DataFrame(raw_data)
        if not df_master.empty:
            df_enriched = enhance_dataframe_with_insights(df_master)
            # Ensure 'outcode' column exists
            if 'outcode' not in df_enriched.columns:
                if 'PostCode' in df_enriched.columns:
                     df_enriched['outcode'] = df_enriched['PostCode'].str.split(' ').str[0]
                elif 'postcode' in df_enriched.columns:
                     df_enriched['outcode'] = df_enriched['postcode'].str.split(' ').str[0]
            
            # Ensure 'manual_review' has valid defaults
            if 'manual_review' not in df_enriched.columns:
                df_enriched['manual_review'] = 'pending'
            else:
                df_enriched['manual_review'] = df_enriched['manual_review'].fillna('pending')

    except Exception as e:
        st.error(f"Error loading initial data: {e}")

    # --- Sidebar Configuration & Filters ---
    with st.sidebar:
        st.header("Configuration")
        # Allow overriding path (will trigger reload on change)
        bq_path_input = st.text_input("BigQuery Table Path", value=bq_path)
        if bq_path_input != bq_path:
            # If path changes, we'd ideally reload. simple st script flow handles this naturally on rerun
            pass

        st.header("Discovery Filters")
        
        # 1. Postcode (Outcode) - Dynamic from Data
        available_outcodes = []
        if not df_enriched.empty and 'outcode' in df_enriched.columns:
            available_outcodes = sorted(df_enriched['outcode'].dropna().unique().tolist())
            
        outcode_filter = st.multiselect("Postcode Area (Outcode)", options=available_outcodes, default=[]) 
        
        # 2. Review Status - Fixed Options per User Request
        manual_review_filter = st.multiselect(
            "Review Status",
            options=["accepted", "rejected", "pending"],
            default=[]
        )

        # 3. AI Filters
        ai_verdict_filter = st.multiselect(
            "AI Verdict",
            ["ACCEPTED", "MAYBE", "REJECTED", "PENDING"],
            default=[]
        )
        
        min_match_score = st.slider("Min Match Score", 0, 100, 0)
        min_auth_score = st.slider("Min Authenticity", 0, 5, 0)

        st.divider()
        # Actions removed (moved to main grid selection)

        st.divider()
        st.header("Data Sync")
        uploaded_file = st.file_uploader("Upload FSA Data (CSV)", type="csv")
        if uploaded_file and st.button("Sync to BigQuery"):
            st.info("Sync triggered...")
            # Sync logic...

    # --- Main Unified Interface ---
    
    if not df_enriched.empty:
        # Apply Filters
        filtered_df = df_enriched.copy()
        
        if outcode_filter:
            filtered_df = filtered_df[filtered_df['outcode'].isin(outcode_filter)]
            
        if manual_review_filter:
             filtered_df = filtered_df[filtered_df['manual_review'].isin(manual_review_filter)]
        
        if "insight_verdict" in filtered_df.columns and ai_verdict_filter:
            filtered_df = filtered_df[filtered_df["insight_verdict"].isin(ai_verdict_filter)]
            
        if "insight_score" in filtered_df.columns and min_match_score > 0:
            filtered_df = filtered_df[filtered_df["insight_score"] >= min_match_score]
            
        if "insight_authenticity" in filtered_df.columns and min_auth_score > 0:
             filtered_df = filtered_df[filtered_df["insight_authenticity"] >= min_auth_score]

        # Metric Summary
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Restaurants", len(df_enriched))
        m2.metric("Filtered View", len(filtered_df))
        m3.metric("AI Profiled", len(filtered_df[filtered_df['detailed_insights'].notna()]))

        st.subheader("Restaurant Registry")
        
        # Main Grid - Multi Selection Mode
        selection_event = display_data(filtered_df, key="main_input_grid")
        selected_rows = get_selected_rows(selection_event, filtered_df)
        
        # Action Bar for Selection
        if selected_rows is not None and not selected_rows.empty:
            st.divider()
            c_act, c_msg = st.columns([1, 4])
            with c_act:
                count = len(selected_rows)
                if st.button(f"Generate Profiles for {count} Selected"):
                    # Extract IDs
                    fhrsids = []
                    if 'fhrsid' in selected_rows.columns:
                        fhrsids = selected_rows['fhrsid'].astype(str).tolist()
                    elif 'FHRSID' in selected_rows.columns:
                        fhrsids = selected_rows['FHRSID'].astype(str).tolist()
                    
                    if fhrsids:
                        with st.spinner(f"Generating profiles for {count} restaurants..."):
                            result_msg = execute_gemini_enrichment(
                                project_id, 
                                dataset_id, 
                                table_id, 
                                fhrsids=fhrsids
                            )
                            st.success(result_msg)
                            time.sleep(2) # Brief pause to read
                            st.rerun()
                    else:
                        st.error("Could not identify FHRSIDs for selection.")
        
        # Deep Dive Section (Integrated below grid)
        if selected_rows is not None and not selected_rows.empty:
            st.markdown("### 🔬 Deep Dive Analysis")
            # Loop
            for idx, row in selected_rows.iterrows():
                render_insights_details(row)
        else:
            st.info("👆 Select one or more restaurants in the table above to view their AI Profile or Generate new ones.")
                    
    else:
        st.warning("No data loaded. Check connection or table path.")

if __name__ == "__main__":
    main()
