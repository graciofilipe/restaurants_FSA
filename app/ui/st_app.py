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

@st.cache_data
def get_cached_outcodes(project_id, dataset_id, table_id):
    return get_distinct_outcodes(project_id, dataset_id, table_id)

def main():
    st.title("🍔 FSA Restaurant Explorer & Profiler")
    
    # Session State Initialization
    if 'df_enriched' not in st.session_state:
        st.session_state.df_enriched = pd.DataFrame()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Config
    bq_path = DEFAULT_BQ_PATH 
    project_id, dataset_id, table_id = bq_path.split('.')
    
    # --- Sidebar Filters (Lazy) ---
    with st.sidebar:
        st.header("Configuration")
        bq_path_input = st.text_input("BigQuery Table Path", value=bq_path)
        
        st.header("Discovery Filters")
        
        # 1. Postcode (Server-Side)
        # Fetch available outcodes from BQ metadata to populate list
        try:
            available_outcodes = get_cached_outcodes(project_id, dataset_id, table_id)
        except Exception as e:
            st.error(f"Failed to fetch outcodes: {e}")
            available_outcodes = []
            
        outcode_filter = st.multiselect("Postcode Area (Outcode)", options=available_outcodes, default=[]) 
        
        # 2. Review Status (Server-Side)
        manual_review_filter = st.multiselect(
            "Review Status",
            options=["accepted", "rejected", "pending"],
            default=["pending"] # Default to pending usually makes sense for workflow
        )

        # 3. AI Filters (Client-Side after load)
        ai_verdict_filter = st.multiselect(
            "AI Verdict",
            ["ACCEPTED", "MAYBE", "REJECTED", "PENDING"],
            default=[]
        )
        
        min_match_score = st.slider("Min Match Score", 0, 100, 0)
        min_auth_score = st.slider("Min Authenticity", 0, 5, 0)

        st.divider()
        # Load Button
        if st.button("Load Data", type="primary"):
            with st.spinner("Fetching data from BigQuery..."):
                try:
                    # Server-side Filtering
                    raw_data = load_filtered_data_from_bq(
                        project_id, 
                        dataset_id, 
                        table_id,
                        review_status_filter=manual_review_filter,
                        postcode_areas=outcode_filter
                    )
                    
                    df_master = pd.DataFrame(raw_data)
                    if not df_master.empty:
                        # Enrichment
                        df_enriched = enhance_dataframe_with_insights(df_master)
                        
                        # Normalize Columns
                        # Ensure 'outcode' column exists
                        if 'outcode' not in df_enriched.columns:
                            if 'PostCode' in df_enriched.columns:
                                df_enriched['outcode'] = df_enriched['PostCode'].str.split(' ').str[0]
                            elif 'postcode' in df_enriched.columns:
                                df_enriched['outcode'] = df_enriched['postcode'].str.split(' ').str[0]
                        
                        # Ensure 'manual_review'
                        if 'manual_review' not in df_enriched.columns:
                            df_enriched['manual_review'] = 'pending'
                        else:
                            df_enriched['manual_review'] = df_enriched['manual_review'].fillna('pending')

                        st.session_state.df_enriched = df_enriched
                        st.session_state.data_loaded = True
                    else:
                        st.session_state.df_enriched = pd.DataFrame()
                        st.session_state.data_loaded = True
                        st.warning("No data found matching criteria.")
                        
                except Exception as e:
                    st.error(f"Error loading data: {e}")

        st.divider()
        st.header("Data Sync")
        uploaded_file = st.file_uploader("Upload FSA Data (CSV)", type="csv")
        if uploaded_file and st.button("Sync to BigQuery"):
            st.info("Sync triggered...")
            # Sync logic...

    # --- Main Interface ---
    if st.session_state.data_loaded and not st.session_state.df_enriched.empty:
        df_display = st.session_state.df_enriched.copy()
        
        # Apply Client-Side AI Filters
        if "insight_verdict" in df_display.columns and ai_verdict_filter:
            df_display = df_display[df_display["insight_verdict"].isin(ai_verdict_filter)]
            
        if "insight_score" in df_display.columns and min_match_score > 0:
            df_display = df_display[df_display["insight_score"] >= min_match_score]
            
        if "insight_authenticity" in df_display.columns and min_auth_score > 0:
             df_display = df_display[df_display["insight_authenticity"] >= min_auth_score]

        # Metric Summary
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Loaded", len(st.session_state.df_enriched))
        m2.metric("Filtered View", len(df_display))
        m3.metric("AI Profiled", len(df_display[df_display['detailed_insights'].notna()]))

        st.subheader("Restaurant Registry")
        
        # Main Grid - Multi Selection Mode
        selection_event = display_data(df_display, key="main_input_grid")
        selected_rows = get_selected_rows(selection_event, df_display)
        
        # Action Bar for Selection
        if selected_rows is not None and not selected_rows.empty:
            st.divider()
            c_act, c_msg = st.columns([1, 4])
            with c_act:
                count = len(selected_rows)
                if st.button(f"Generate Profiles for {count} Selected"):
                    # Extract IDs
                    fhrsids = []
                    # Robust ID extraction
                    col_map = {c.lower(): c for c in selected_rows.columns}
                    id_col = col_map.get('fhrsid')
                    
                    if id_col:
                        fhrsids = selected_rows[id_col].astype(str).tolist()
                    
                    if fhrsids:
                        with st.spinner(f"Generating profiles for {count} restaurants..."):
                            result_msg = execute_gemini_enrichment(
                                project_id, 
                                dataset_id, 
                                table_id, 
                                fhrsids=fhrsids
                            )
                            st.success(result_msg)
                            time.sleep(2)
                            # Invalidate cache/reload? 
                            # For now, just rerun which might re-render state. 
                            # Ideally we reload data but that's expensive.
                            st.rerun()
                    else:
                        st.error("Could not identify FHRSIDs for selection.")
        
        # Deep Dive Section
        if selected_rows is not None and not selected_rows.empty:
            st.markdown("### 🔬 Deep Dive Analysis")
            for idx, row in selected_rows.iterrows():
                render_insights_details(row)
        else:
            st.info("👆 Select one or more restaurants in the table above to view their AI Profile or Generate new ones.")
    
    elif st.session_state.data_loaded and st.session_state.df_enriched.empty:
        st.warning("No data found. Try adjusting filters and clicking 'Load Data'.")
    else:
        st.info("👈 Use the filters in the sidebar and click **Load Data** to begin.")

if __name__ == "__main__":
    main()
