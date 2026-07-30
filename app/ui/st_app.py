import streamlit as st
import time
import pandas as pd
from app.services.bq_utils import (
    load_all_data_from_bq, 
    execute_gemini_enrichment, 
    get_distinct_local_authorities,
    get_distinct_outcodes,
    load_filtered_data_from_bq,
    bulk_update_reviews,
    update_rows_in_bigquery
)
from app.services.ml_prediction import generate_predictions
from app.core.data_processing import enhance_dataframe_with_insights

st.set_page_config(page_title="FSA Restaurant Explorer", layout="wide")

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

DISPLAY_COLUMNS = [
    "fhrsid", "businessname", "in_scope", "rating_source", "user_rating", "predicted_user_rating",
    "addressline1", "addressline2", "addressline3", 
    "postcode", "localauthorityname", "first_seen", "manual_review",
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
    biz_name = row.get('BusinessName') or row.get('businessname') or "Unknown Restaurant"
    st.markdown(f"### 🍽️ Profiling: {biz_name}")
    
    insights = row.get('detailed_insights')
    if not insights or not isinstance(insights, dict):
        st.info("No detailed persona profile available for this restaurant yet.")
        if row.get('insight_summary') and isinstance(row.get('insight_summary'), str):
             st.write(f"**Legacy Summary:** {row['insight_summary']}")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Persona Match Score", f"{insights.get('match_score', 'N/A')}/100")
        
    with col2:
        if insights.get('summary_reasoning'):
            st.info(f"**Verdict:** {insights['summary_reasoning']}")

    st.markdown("---")
    st.markdown("#### 🔬 The 6-Pillar Analysis")
    
    pillars = [k for k in insights.keys() if k and k[0].isdigit()]
    pillars.sort()
    
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
    in_scope_filter, 
    outcode_filter, 
    first_seen_start_date=None,
    local_authority_filter=None
):
    """
    Helper to load data into session state.
    """
    with st.spinner("Fetching data from BigQuery..."):
        try:
            raw_data = load_filtered_data_from_bq(
                project_id, 
                dataset_id, 
                table_id,
                in_scope_filter=in_scope_filter,
                postcode_areas=outcode_filter,
                first_seen_start_date=first_seen_start_date,
                local_authority_filter=local_authority_filter
            )
            
            df_master = pd.DataFrame(raw_data)
            if not df_master.empty:
                df_enriched = enhance_dataframe_with_insights(df_master)
                
                if 'outcode' not in df_enriched.columns:
                    if 'PostCode' in df_enriched.columns:
                        df_enriched['outcode'] = df_enriched['PostCode'].str.split(' ').str[0]
                    elif 'postcode' in df_enriched.columns:
                        df_enriched['outcode'] = df_enriched['postcode'].str.split(' ').str[0]
                
                if 'in_scope' not in df_enriched.columns:
                    df_enriched['in_scope'] = None

                st.session_state.df_enriched = df_enriched
                st.session_state.data_loaded = True
            else:
                st.session_state.df_enriched = pd.DataFrame()
                st.session_state.data_loaded = True
                st.warning("No data found matching criteria.")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")

def main():
    st.title("🍔 FSA Restaurant Explorer & Scoring Engine")
    
    if 'df_enriched' not in st.session_state:
        st.session_state.df_enriched = pd.DataFrame()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    bq_path = DEFAULT_BQ_PATH 
    project_id, dataset_id, table_id = bq_path.split('.')
    
    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("Configuration")
        bq_path_input = st.text_input("BigQuery Table Path", value=bq_path)
        
        st.header("Scope & Location Filters")
        
        scope_options_map = {
            "In Scope (Restaurants)": "in_scope",
            "Out of Scope (Cafes/Bakeries)": "out_of_scope",
            "Unprocessed / Needs Triage": "unprocessed"
        }
        
        selected_scope_labels = st.multiselect(
            "Establishment Scope",
            options=list(scope_options_map.keys()),
            default=["In Scope (Restaurants)", "Unprocessed / Needs Triage"]
        )
        in_scope_filter_values = [scope_options_map[lbl] for lbl in selected_scope_labels]
        
        try:
            available_outcodes = get_cached_outcodes(project_id, dataset_id, table_id)
        except Exception as e:
            st.error(f"Failed to fetch outcodes: {e}")
            available_outcodes = []
            
        outcode_filter = st.multiselect("Postcode Area (Outcode)", options=available_outcodes, default=[]) 

        try:
            available_authorities = get_cached_local_authorities(project_id, dataset_id, table_id)
        except Exception as e:
            st.error(f"Failed to fetch local authorities: {e}")
            available_authorities = []
        
        local_authority_filter = st.multiselect("Local Authority", options=available_authorities, default=[])

        st.divider()
        st.write("### 📅 Date Filter")
        first_seen_date = st.date_input(
            "First Seen After",
            value=None,
            max_value=pd.Timestamp.now().date(),
            help="Load restaurants first seen ON or AFTER this date."
        )

        if st.button("Load Data", type="primary"):
            load_data_into_state(
                project_id, 
                dataset_id, 
                table_id, 
                in_scope_filter_values, 
                outcode_filter,
                first_seen_start_date=first_seen_date,
                local_authority_filter=local_authority_filter
            )

    # --- Main Interface ---
    if st.session_state.data_loaded and not st.session_state.df_enriched.empty:
        df_display = st.session_state.df_enriched.copy()
        
        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Loaded", len(df_display))
        m2.metric("In-Scope", len(df_display[df_display['in_scope'] == True]) if 'in_scope' in df_display.columns else 0)
        m3.metric("User Rated", len(df_display[df_display['user_rating'].notna()]) if 'user_rating' in df_display.columns else 0)
        m4.metric("ML Predicted", len(df_display[df_display['predicted_user_rating'].notna()]) if 'predicted_user_rating' in df_display.columns else 0)

        tab_triage, tab_rating, tab_predictions = st.tabs(["📥 1. Scope Triage Inbox", "✍️ 2. Desk & Visit Rating Hub", "🤖 3. ML Predictions & Discovery"])
        
        # -------------------------------------------------------------
        # TAB 1: SCOPE TRIAGE INBOX
        # -------------------------------------------------------------
        with tab_triage:
            st.subheader("Scope Triage Inbox")
            st.caption("Classify new or existing establishments into In-Scope (Restaurants) vs Out-of-Scope (Cafes, Bakeries, Supermarkets).")
            
            selection_event_triage = display_data(df_display, key="triage_grid")
            selected_triage = get_selected_rows(selection_event_triage, df_display)
            
            if selected_triage is not None and not selected_triage.empty:
                st.divider()
                count = len(selected_triage)
                col_map = {c.lower(): c for c in selected_triage.columns}
                id_col = col_map.get('fhrsid')
                
                st.write(f"**Batch Scope Triage for {count} Selected Establishments:**")
                c_in, c_out, c_reset = st.columns(3)
                
                if c_in.button("✅ Mark as In-Scope (Restaurant)"):
                    if id_col:
                        ids = selected_triage[id_col].astype(str).tolist()
                        df_up = pd.DataFrame({'fhrsid': ids, 'in_scope': [True] * len(ids)})
                        with st.spinner(f"Updating {len(ids)} rows to In-Scope..."):
                            success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                            if success:
                                st.success(msg)
                                load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                st.rerun()
                            else:
                                st.error(msg)
                                
                if c_out.button("🚫 Mark as Out-of-Scope (Bakery/Cafe)"):
                    if id_col:
                        ids = selected_triage[id_col].astype(str).tolist()
                        df_up = pd.DataFrame({'fhrsid': ids, 'in_scope': [False] * len(ids)})
                        with st.spinner(f"Updating {len(ids)} rows to Out-of-Scope..."):
                            success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                            if success:
                                st.success(msg)
                                load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                st.rerun()
                            else:
                                st.error(msg)

                if c_reset.button("🔄 Reset to Unprocessed"):
                    if id_col:
                        ids = selected_triage[id_col].astype(str).tolist()
                        df_up = pd.DataFrame({'fhrsid': ids, 'in_scope': [None] * len(ids)})
                        with st.spinner(f"Resetting {len(ids)} rows..."):
                            success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                            if success:
                                st.success(msg)
                                load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                st.rerun()
                            else:
                                st.error(msg)

        # -------------------------------------------------------------
        # TAB 2: DESK & VISIT RATING HUB
        # -------------------------------------------------------------
        with tab_rating:
            st.subheader("Desk & Visit Rating Hub (1 to 10 Scale)")
            st.caption("Assign user scores (1-10) for in-scope restaurants from desk evaluation (unvisited) or post-visit ground truth.")
            
            # Filter in-scope items for rating
            df_in_scope = df_display[df_display['in_scope'] == True] if 'in_scope' in df_display.columns else df_display.copy()
            
            rating_view = st.radio(
                "Filter View",
                options=["All In-Scope", "Unrated Only (Desk Rating Opportunities)", "User Rated Only"],
                horizontal=True
            )
            
            if rating_view == "Unrated Only (Desk Rating Opportunities)" and "user_rating" in df_in_scope.columns:
                df_rating_view = df_in_scope[df_in_scope["user_rating"].isna()]
            elif rating_view == "User Rated Only" and "user_rating" in df_in_scope.columns:
                df_rating_view = df_in_scope[df_in_scope["user_rating"].notna()]
            else:
                df_rating_view = df_in_scope.copy()
                
            selection_event_rating = display_data(df_rating_view, key="rating_grid")
            selected_rating_rows = get_selected_rows(selection_event_rating, df_rating_view)
            
            if selected_rating_rows is not None and not selected_rating_rows.empty:
                st.divider()
                st.write(f"### ⭐ Score {len(selected_rating_rows)} Selected Establishment(s)")
                
                col_score, col_source, col_btn = st.columns([2, 2, 2])
                with col_score:
                    new_rating = st.slider("User Score (1 to 10)", min_value=1, max_value=10, value=7, step=1)
                with col_source:
                    rating_source_type = st.selectbox("Rating Source", options=["desk", "visited"], index=0)
                with col_btn:
                    st.write("")
                    st.write("")
                    if st.button("Submit Score to BigQuery", type="primary"):
                        col_map = {c.lower(): c for c in selected_rating_rows.columns}
                        id_col = col_map.get('fhrsid')
                        if id_col:
                            ids = selected_rating_rows[id_col].astype(str).tolist()
                            df_up = pd.DataFrame({
                                'fhrsid': ids,
                                'user_rating': [new_rating] * len(ids),
                                'rating_source': [rating_source_type] * len(ids),
                                'in_scope': [True] * len(ids)
                            })
                            with st.spinner(f"Saving score {new_rating}/10 for {len(ids)} restaurants..."):
                                success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                                if success:
                                    st.success(f"Saved: {msg}")
                                    load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                    st.rerun()
                                else:
                                    st.error(msg)
                                    
                # Single-row detailed 6-pillar view
                if len(selected_rating_rows) == 1:
                    render_insights_details(selected_rating_rows.iloc[0])

        # -------------------------------------------------------------
        # TAB 3: ML PREDICTIONS & DISCOVERY
        # -------------------------------------------------------------
        with tab_predictions:
            st.subheader("🤖 ML Preference Predictions & Discovery")
            st.caption("Sort and filter restaurants by continuous predicted preference score (1 to 10 scale).")
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                show_pred_only = st.checkbox("Only Show Restaurants with ML Predictions", value=True)
            with col_p2:
                min_pred_score = st.slider("Minimum Predicted Rating", min_value=1.0, max_value=10.0, value=6.0, step=0.5)
                
            df_pred_view = df_display.copy()
            if "predicted_user_rating" in df_pred_view.columns:
                if show_pred_only:
                    df_pred_view = df_pred_view[df_pred_view["predicted_user_rating"].notna()]
                df_pred_view = df_pred_view[df_pred_view["predicted_user_rating"] >= min_pred_score]
                df_pred_view = df_pred_view.sort_values(by="predicted_user_rating", ascending=False)
                
            selection_event_pred = display_data(df_pred_view, key="pred_grid")
            selected_pred_rows = get_selected_rows(selection_event_pred, df_pred_view)
            
            st.divider()
            st.write("### ⚙️ ML Operations")
            col_gen_p, col_train_p = st.columns(2)
            
            with col_gen_p:
                st.write("**Generate Predictions for Unrated Restaurants:**")
                force_maps = st.checkbox("Force Regenerate Maps Data", key="force_maps_p")
                force_gemini = st.checkbox("Force Regenerate Gemini Profiles", key="force_gemini_p")
                
                target_count = len(selected_pred_rows) if (selected_pred_rows is not None and not selected_pred_rows.empty) else 50
                btn_label = f"Generate Predictions ({target_count} selected)" if (selected_pred_rows is not None and not selected_pred_rows.empty) else "Generate Batch Predictions (Top 50 Unrated)"
                
                if st.button(btn_label):
                    fhrsids = None
                    if selected_pred_rows is not None and not selected_pred_rows.empty:
                        col_map = {c.lower(): c for c in selected_pred_rows.columns}
                        id_col = col_map.get('fhrsid')
                        if id_col:
                            fhrsids = selected_pred_rows[id_col].astype(str).tolist()
                    
                    with st.spinner("Generating ML predictions..."):
                        success, msg = generate_predictions(
                            project_id, dataset_id, table_id,
                            "restaurant_preference_model",
                            limit=50,
                            target_fhrsids=fhrsids,
                            force_maps=force_maps,
                            force_gemini=force_gemini
                        )
                        if success:
                            st.success(msg)
                            load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                            st.rerun()
                        else:
                            st.error(msg)
                            
            with col_train_p:
                st.write("**Train BQML Boosted Tree Model:**")
                st.caption("Trains regression model using all in-scope rated restaurants (`user_rating` 1-10).")
                
                if "training_lock" not in st.session_state:
                    st.session_state.training_lock = False
                    
                if st.button("Train BQML Model (Async)", disabled=st.session_state.training_lock, key="btn_train_model"):
                    try:
                        from scripts.train_bqml_model import train_model
                        with st.spinner("Starting BQML model training..."):
                            job_id = train_model(
                                project_id=project_id,
                                dataset_id=dataset_id,
                                table_id=table_id,
                                model_name="restaurant_preference_model",
                                run_async=True
                            )
                            st.success(f"Started model training. Job ID: {job_id}")
                    except Exception as e:
                        st.error(f"Failed to start training: {e}")

    elif st.session_state.data_loaded and st.session_state.df_enriched.empty:
        st.warning("No data found. Try adjusting filters in the sidebar and clicking 'Load Data'.")
    else:
        st.info("👈 Select your Scope & Location filters in the sidebar and click **Load Data** to start.")

if __name__ == "__main__":
    main()
