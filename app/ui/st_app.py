import streamlit as st
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

def display_data(df):
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        hide_index=True,
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
    st.markdown(f"### 🍽️ Profiling: {row['BusinessName']}")
    
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
    # Find keys that start with a digit
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
                # Try to extract score/rating for a prominent display
                score_val = pillar_data.get('score') or pillar_data.get('rating')
                
                c_score, c_details = st.columns([1, 4])
                
                with c_score:
                    if score_val:
                        st.metric("Score", f"{score_val}/5")
                    
                with c_details:
                    # Render other fields
                    for k, v in pillar_data.items():
                        if k not in ['score', 'rating']:
                            # Format key nicely
                            nice_key = k.replace('_', ' ').capitalize()
                            st.write(f"**{nice_key}:** {v}")
            else:
                st.write(pillar_data)

def main():
    st.title("🍔 FSA Restaurant Explorer & Profiler")
    
    # Sidebar Config
    with st.sidebar:
        st.header("Configuration")
        bq_path = st.text_input("BigQuery Table Path", value=DEFAULT_BQ_PATH)
        
        st.header("Discovery Filters")
        
        # New Filters for AI attributes
        ai_verdict_filter = st.multiselect(
            "AI Verdict",
            ["ACCEPTED", "MAYBE", "REJECTED", "PENDING"],
            default=[]
        )
        
        min_match_score = st.slider(
            "Min Match Score",
            0, 100, 0
        )
        
        min_auth_score = st.slider(
            "Min Authenticity",
            0, 5, 0
        )

        st.header("Data Sync")
        uploaded_file = st.file_uploader("Upload FSA Data (CSV)", type="csv")
        
        if uploaded_file and st.button("Sync to BigQuery"):
            try:
                # Basic Sync Trigger (Implementation simplified for restoration)
                # Valid coordinates would theoretically come from CSV
                st.info("Sync triggered... (Coord logic implicit for now)")
                csv_df = load_data_from_csv(uploaded_file)
                project_id, dataset_id, table_id = bq_path.split('.')
                # We need valid coords to run full sync, but assuming CSV has them or we just load CSV?
                # For now using simplified placeholder or minimal logic if csv_df has coords
                pass 
            except Exception as e:
                st.error(f"Sync failed: {e}")

    # Main Tabs
    tab_data, tab_gemini = st.tabs(["Data Overview", "AI Profiling Lab"])
    
    # Global Data Load
    project_id, dataset_id, table_id = bq_path.split('.')
    
    # We should optimize to load filtered data if possible, but for now load all or cached
    # Ideally use st.cache_data in real app context
    try:
        # Load Raw Data
        raw_data = load_all_data_from_bq(project_id, dataset_id, table_id)
        df_master = pd.DataFrame(raw_data)
        
        if not df_master.empty:
            # ENRICH
            df_enriched = enhance_dataframe_with_insights(df_master)
            
            # FILTER
            filtered_df = df_enriched.copy()
            
            if "insight_verdict" in filtered_df.columns and ai_verdict_filter:
                filtered_df = filtered_df[filtered_df["insight_verdict"].isin(ai_verdict_filter)]
                
            if "insight_score" in filtered_df.columns and min_match_score > 0:
                filtered_df = filtered_df[filtered_df["insight_score"] >= min_match_score]
                
            if "insight_authenticity" in filtered_df.columns and min_auth_score > 0:
                 filtered_df = filtered_df[filtered_df["insight_authenticity"] >= min_auth_score]

            with tab_data:
                st.subheader(f"Restaurant Registry ({len(filtered_df)} records)")
                display_data(filtered_df)
                
            with tab_gemini:
                st.subheader("🤖 Culinary Anthropologist (Agent)")
                
                c_action, c_info = st.columns([1, 2])
                with c_action:
                    if st.button("Generate Profiles for Recents"):
                        with st.spinner("Agent calling Vertex AI..."):
                            result_msg = execute_gemini_enrichment(project_id, dataset_id, table_id)
                            st.success(result_msg)
                            st.rerun()
                            
                st.markdown("### Deep Dive View")
                st.write("Select a restaurant in the table below to see the extensive 6-pillar analysis.")
                
                selection_event = display_data(filtered_df)
                selected_rows = get_selected_rows(selection_event, filtered_df)
                
                if selected_rows is not None and not selected_rows.empty:
                    # Show details for the first selected row (or all, but detail view usually 1)
                    for idx, row in selected_rows.iterrows():
                        render_insights_details(row)
                        
        else:
            st.warning("No data found in BigQuery.")
            
    except Exception as e:
        st.error(f"Error loading application: {e}")

if __name__ == "__main__":
    main()
