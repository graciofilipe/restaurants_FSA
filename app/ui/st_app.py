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

# ... (omitting unchanged parts)

            with tab_data:
                st.subheader(f"Restaurant Registry ({len(filtered_df)} records)")
                display_data(filtered_df, key="main_grid")
                
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
                
                selection_event = display_data(filtered_df, key="profile_selector")
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
