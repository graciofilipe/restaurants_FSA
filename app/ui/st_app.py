import streamlit as st
import time
import pandas as pd
from app.services.bq_utils import (
    load_all_data_from_bq, 
    execute_gemini_enrichment, 
    get_distinct_local_authorities,
    get_distinct_outcodes,
    load_filtered_data_from_bq,
    bulk_update_reviews
)
from app.services.ml_prediction import generate_predictions
from app.core.data_processing import enhance_dataframe_with_insights

st.set_page_config(page_title="FSA Restaurant Explorer", layout="wide")

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

DISPLAY_COLUMNS = [
    "fhrsid", "businessname", "addressline1", "addressline2", "addressline3", 
    "postcode", "localauthorityname", "first_seen", "manual_review", "user_rating", "predicted_user_rating",
    "price_level", "maps_rating", "maps_reviews",
    "latitude", "longitude", "maps_url", "business_status", "website_url", "maps_types",
    "gemini_insights", "gemini_insights_structured",
    "match_score",
    "1_value_and_volume_rating", "1_value_and_volume_verdict",
    "2_demographic_community_score", "2_demographic_community_evidence",
    "3_linguistic_signal_score", "3_linguistic_signal_menu_type",
    "4_geographic_precision_region_identified", "4_geographic_precision_specificity_level",
    "5_culinary_uncompromisingness_score", "5_culinary_uncompromisingness_pander_check",
    "6_establishment_integrity_is_sit_down_restaurant", "6_establishment_integrity_type",
    "summary_reasoning"
]

def display_data(df, key=None):
    # Ensure columns exist to avoid Streamlit warnings, though column_order usually handles missing gracefully (hides them)
    # But filters/logic might define columns that are not yet in DF if filtered empty?
    # We pass column_order directly.
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        hide_index=True,
        column_order=DISPLAY_COLUMNS,
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
    
    insights = row.get('detailed_insights')
    if not insights or not isinstance(insights, dict):
        st.info("No detailed persona profile available for this restaurant yet.")
        if row.get('insight_summary') and isinstance(row.get('insight_summary'), str):
             st.write(f"**Legacy Summary:** {row['insight_summary']}")
        return

    
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

@st.cache_data
def get_cached_local_authorities(project_id, dataset_id, table_id):
    return get_distinct_local_authorities(project_id, dataset_id, table_id)

def load_data_into_state(
    project_id, 
    dataset_id, 
    table_id, 
    review_status_filter, 
    outcode_filter, 
    first_seen_start_date=None,
    local_authority_filter=None
):
    """
    Helper to load data into session state.
    Used by 'Load Data' button and after 'Generate Profiles' to ensure freshness.
    """
    with st.spinner("Fetching data from BigQuery..."):
        try:
            # Server-side Filtering
            raw_data = load_filtered_data_from_bq(
                project_id, 
                dataset_id, 
                table_id,
                review_status_filter=review_status_filter,
                postcode_areas=outcode_filter,
                first_seen_start_date=first_seen_start_date,
                local_authority_filter=local_authority_filter
            )
            
            df_master = pd.DataFrame(raw_data)
            if not df_master.empty:
                # Enrichment
                df_enriched = enhance_dataframe_with_insights(df_master)
                
                # Normalize Columns
                if 'outcode' not in df_enriched.columns:
                    if 'PostCode' in df_enriched.columns:
                        df_enriched['outcode'] = df_enriched['PostCode'].str.split(' ').str[0]
                    elif 'postcode' in df_enriched.columns:
                        df_enriched['outcode'] = df_enriched['postcode'].str.split(' ').str[0]
                
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
        try:
            available_outcodes = get_cached_outcodes(project_id, dataset_id, table_id)
        except Exception as e:
            st.error(f"Failed to fetch outcodes: {e}")
            available_outcodes = []
            
        outcode_filter = st.multiselect("Postcode Area (Outcode)", options=available_outcodes, default=[]) 

        # 2. Local Authority (Server-Side)
        try:
            available_authorities = get_cached_local_authorities(project_id, dataset_id, table_id)
        except Exception as e:
            st.error(f"Failed to fetch local authorities: {e}")
            available_authorities = []
        
        local_authority_filter = st.multiselect("Local Authority", options=available_authorities, default=[])
        
        # 3. Review Status (Server-Side)
        manual_review_filter = st.multiselect(
            "Manual Review",
            options=["accepted", "rejected", "pending", "not reviewed"],
            default=["pending"]
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
        st.write("### 🤖 ML Prediction Filters")
        show_predicted_only = st.checkbox("Only show restaurants with ML Predictions")

        if "training_lock" not in st.session_state:
            st.session_state.training_lock = False
            
        if st.button("Train BQML Model (Async)", disabled=st.session_state.training_lock):
            st.session_state.trigger_training = True
            st.session_state.training_lock = True
            st.rerun()

        if st.session_state.get('trigger_training', False):
            st.session_state.trigger_training = False
            try:
                from scripts.train_bqml_model import train_model
                with st.spinner("Executing JIT check and starting training..."):
                    job_id = train_model(
                        project_id=project_id,
                        dataset_id=dataset_id,
                        table_id=table_id,
                        model_name="restaurant_preference_model",
                        run_async=True
                    )
                    st.success(f"Started training. Job ID: {job_id}")
            except Exception as e:
                st.error(f"Failed to start training: {e}")
            finally:
                st.session_state.training_lock = False
                
        st.divider()
        
        # 4. Date Filter (Server-Side)
        st.write("### 📅 Date Filter")
        first_seen_date = st.date_input(
            "First Seen After",
            value=None,
            min_value=None,
            max_value=pd.Timestamp.now().date(),
            help="Load restaurants first seen ON or AFTER this date."
        )

        
        # Load Button
        if st.button("Load Data", type="primary"):
            load_data_into_state(
                project_id, 
                dataset_id, 
                table_id, 
                manual_review_filter, 
                outcode_filter,
                first_seen_start_date=first_seen_date,
                local_authority_filter=local_authority_filter
            )



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
             
        if show_predicted_only and "predicted_user_rating" in df_display.columns:
            df_display = df_display[df_display["predicted_user_rating"].notna()]

        # Metric Summary
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Loaded", len(st.session_state.df_enriched))
        m2.metric("Filtered View", len(df_display))
        m3.metric("AI Profiled", len(df_display[df_display['detailed_insights'].notna()]))

        tab_explore, tab_bulk = st.tabs(["🔍 Explore & Profile", "⭐ Bulk Rating Mode"])
        
        with tab_explore:
            st.subheader("Restaurant Registry")
            
            # Main Grid - Multi Selection Mode
            selection_event = display_data(df_display, key="main_input_grid")
            selected_rows = get_selected_rows(selection_event, df_display)
            
            # Action Bar for Selection
            if selected_rows is not None and not selected_rows.empty:
                st.divider()
                
                # Action Containers
                col_gen, col_predict, col_update = st.columns([1, 1, 2])
                
                # 1. Generate Profiles
                with col_gen:
                    count = len(selected_rows)
                    if st.button(f"Generate Profiles for {count} Selected"):
                        # Extract IDs
                        fhrsids = []
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
                                time.sleep(1) # Brief pause for user to see success
                                
                                # Reload data to ensure freshness (fix for stale UI)
                                load_data_into_state(
                                    project_id, 
                                    dataset_id, 
                                    table_id, 
                                    manual_review_filter, 
                                    outcode_filter,
                                    first_seen_start_date=first_seen_date,
                                    local_authority_filter=local_authority_filter
                                )
                                
                                st.rerun()
                        else:
                            st.error("Could not identify FHRSIDs for selection.")

                # 1.5 Generate Predictions
                with col_predict:
                    count = len(selected_rows)
                    st.write("Prediction Options:")
                    force_maps = st.checkbox("Force Regenerate Maps Data", key="force_maps_chk")
                    force_gemini = st.checkbox("Force Regenerate Gemini Profiles", key="force_gemini_chk")
                    if st.button(f"Generate Predictions for {count} Selected"):
                        fhrsids = []
                        col_map = {c.lower(): c for c in selected_rows.columns}
                        id_col = col_map.get('fhrsid')
                        
                        if id_col:
                            fhrsids = selected_rows[id_col].astype(str).tolist()
                        
                        if fhrsids:
                            with st.spinner(f"Generating predictions for {count} restaurants..."):
                                try:
                                    success, msg = generate_predictions(
                                        project_id, 
                                        dataset_id, 
                                        table_id, 
                                        "restaurant_preference_model", 
                                        target_fhrsids=fhrsids,
                                        force_maps=force_maps,
                                        force_gemini=force_gemini
                                    )
                                    if success:
                                        st.success(msg)
                                        time.sleep(1)
                                        load_data_into_state(
                                            project_id, 
                                            dataset_id, 
                                            table_id, 
                                            manual_review_filter, 
                                            outcode_filter,
                                            first_seen_start_date=first_seen_date,
                                            local_authority_filter=local_authority_filter
                                        )
                                        st.rerun()
                                    else:
                                        st.error(msg)
                                except Exception as e:
                                    st.error(f"Prediction failed: {e}")
                        else:
                            st.error("Could not identify FHRSIDs for selection.")

                # 2. Update Status
                with col_update:
                    st.write("Update Manual Review Status:")
                    c_acc, c_rej, c_pend, c_reset = st.columns(4)
                    
                    new_status = None
                    if c_acc.button("✅ Accept"):
                        new_status = "accepted"
                    if c_rej.button("❌ Reject"):
                        new_status = "rejected"
                    if c_pend.button("⏳ Pending"):
                        new_status = "pending"
                    if c_reset.button("🔄 Reset"):
                        new_status = "not reviewed"
                    
                    if new_status:
                        # Extract IDs
                        col_map = {c.lower(): c for c in selected_rows.columns}
                        id_col = col_map.get('fhrsid')
                        
                        if id_col:
                            ids_to_update = selected_rows[id_col].unique().tolist()
                            
                            # Prepare update DataFrame
                            df_updates = pd.DataFrame({
                                'fhrsid': ids_to_update,
                                'manual_review': [new_status] * len(ids_to_update)
                            })
                            
                            with st.spinner(f"Updating {len(ids_to_update)} rows to '{new_status}'..."):
                                success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_updates)
                                if success:
                                    st.success(f"Updated: {msg}")
                                    # Update session state directly to reflect change immediately
                                    if 'df_enriched' in st.session_state:
                                        df_s = st.session_state.df_enriched
                                        s_col_map = {c.lower(): c for c in df_s.columns}
                                        s_id_col = s_col_map.get('fhrsid')
                                        
                                        if s_id_col:
                                             mask = df_s[s_id_col].astype(str).isin([str(x) for x in ids_to_update])
                                             if 'manual_review' in df_s.columns:
                                                 st.session_state.df_enriched.loc[mask, 'manual_review'] = new_status
                                        
                                        time.sleep(1)
                                        st.rerun()
                                else:
                                    st.error(f"Update failed: {msg}")
                        else:
                             st.error("Could not identify FHRSID for updates.")
        
                # 3. Individual User Rating (Only if 1 row selected)
                if len(selected_rows) == 1:
                    st.divider()
                    st.write("### ⭐ Rate Restaurant")
                    row = selected_rows.iloc[0]
                    col_map = {c.lower(): c for c in row.index}
                    id_col = col_map.get('fhrsid')
                    biz_name = row.get('BusinessName') or row.get('businessname') or "Unknown Restaurant"

                    pred_rating = row.get('predicted_user_rating')
                    if pd.notna(pred_rating):
                        st.info(f"🤖 **ML Predicted Rating:** {pred_rating:.1f} / 10")

                    current_rating = row.get('user_rating')
                    if pd.isna(current_rating):
                        current_rating = 5

                    user_rating = st.slider(f"Your Rating for {biz_name} (1-10)", min_value=1, max_value=10, value=int(current_rating), step=1)
                    if st.button("Submit Rating", type="primary"):
                        if id_col:
                            fhrsid_to_update = str(row[id_col])
                            with st.spinner("Saving rating..."):
                                from app.services.bq_utils import update_rows_in_bigquery
                                success = update_rows_in_bigquery(
                                    project_id, dataset_id, table_id,
                                    fhrsid_to_update,
                                    {'user_rating': user_rating}
                                )
                                if success:
                                    st.success("Rating saved!")
                                    if 'df_enriched' in st.session_state:
                                        df_s = st.session_state.df_enriched
                                        s_col_map = {c.lower(): c for c in df_s.columns}
                                        s_id_col = s_col_map.get('fhrsid')
                                        if s_id_col:
                                            mask = df_s[s_id_col].astype(str) == fhrsid_to_update
                                            if 'user_rating' not in df_s.columns:
                                                st.session_state.df_enriched['user_rating'] = None

                                            st.session_state.df_enriched.loc[mask, 'user_rating'] = user_rating
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to save rating.")
                        else:
                            st.error("Could not identify FHRSID.")
            
            # Deep Dive Section
            if selected_rows is not None and not selected_rows.empty:
                st.markdown("### 🔬 Deep Dive Analysis")
                for idx, row in selected_rows.iterrows():
                    render_insights_details(row)
            else:
                st.info("👆 Select one or more restaurants in the table above to view their AI Profile or Generate new ones.")

        with tab_bulk:
            st.write("### Bulk Rating Editor")
            st.info("Edit user ratings directly below. Click `Save Bulk Ratings` when done to persist changes to BigQuery.")
            
            bulk_cols = ["fhrsid", "businessname", "addressline1", "postcode", "localauthorityname", "manual_review", "user_rating"]
            available_bulk_cols = [c for c in bulk_cols if c in df_display.columns]
            
            if "user_rating" not in available_bulk_cols:
                df_display["user_rating"] = pd.NA
                available_bulk_cols.append("user_rating")

            df_bulk_reset = df_display[available_bulk_cols].reset_index(drop=True)
            
            edited_df = st.data_editor(
                df_bulk_reset,
                use_container_width=True,
                hide_index=True,
                disabled=[c for c in available_bulk_cols if c != "user_rating"],
                column_config={
                    "user_rating": st.column_config.NumberColumn(
                        "User Rating",
                        min_value=1,
                        max_value=10,
                        step=1,
                        help="Rate from 1 to 10"
                    )
                },
                key="bulk_editor"
            )
            
            if st.button("Save Bulk Ratings", type="primary"):
                # Handle NA correctly using fillna to a placeholder for comparison
                mask = edited_df["user_rating"] != df_bulk_reset["user_rating"]
                mask = mask & ~(edited_df["user_rating"].isna() & df_bulk_reset["user_rating"].isna())
                
                updates = edited_df[mask].copy()
                
                if not updates.empty:
                    with st.spinner(f"Saving {len(updates)} ratings..."):
                        if "manual_review" not in updates.columns:
                            updates["manual_review"] = "pending"
                        
                        df_to_update = updates[["fhrsid", "manual_review", "user_rating"]].copy()
                        success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_to_update)
                        
                        if success:
                            st.success(f"Updated: {msg}")
                            if "df_enriched" in st.session_state:
                                df_s = st.session_state.df_enriched
                                s_col_map = {c.lower(): c for c in df_s.columns}
                                s_id_col = s_col_map.get("fhrsid")
                                
                                if s_id_col:
                                    if "user_rating" not in df_s.columns:
                                        df_s["user_rating"] = pd.NA
                                    for idx, row in df_to_update.iterrows():
                                        update_mask = df_s[s_id_col].astype(str) == str(row["fhrsid"])
                                        st.session_state.df_enriched.loc[update_mask, "user_rating"] = row["user_rating"]
                            
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Update failed: {msg}")
                else:
                    st.info("No rating changes detected.")
    elif st.session_state.data_loaded and st.session_state.df_enriched.empty:
        st.warning("No data found. Try adjusting filters and clicking 'Load Data'.")
    else:
        st.info("👈 Use the filters in the sidebar and click **Load Data** to begin.")

if __name__ == "__main__":
    main()
