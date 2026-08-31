import streamlit as st
import pandas as pd
from app.services.bq_utils import (
    get_distinct_local_authorities,
    get_distinct_outcodes,
    load_filtered_data_from_bq,
    bulk_update_reviews,
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
    "gemini_insights_structured",
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


def filter_and_sort_restaurants(
    df: pd.DataFrame,
    scope_filter: str = "All Loaded",
    rating_filter: str = "All",
    pred_filter: str = "All",
    min_pred_score: float = 1.0,
    search_query: str = "",
    sort_by: str = "Predicted Rating (High to Low)",
) -> pd.DataFrame:
    """
    Applies in-memory filtering and sorting to the restaurant DataFrame.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # 1. Scope Filter
    if "in_scope" in filtered.columns:
        if scope_filter == "In-Scope (Restaurants)":
            filtered = filtered[filtered["in_scope"] == True]
        elif scope_filter == "Out-of-Scope":
            filtered = filtered[filtered["in_scope"] == False]
        elif scope_filter == "Unprocessed / Triage":
            filtered = filtered[filtered["in_scope"].isna()]

    # 2. Rating Status Filter
    if "user_rating" in filtered.columns:
        if rating_filter == "Unrated Only":
            filtered = filtered[filtered["user_rating"].isna()]
        elif rating_filter == "User Rated Only":
            filtered = filtered[filtered["user_rating"].notna()]

    # 3. ML Prediction Filter
    if "predicted_user_rating" in filtered.columns:
        if pred_filter == "Predicted Only":
            filtered = filtered[
                filtered["predicted_user_rating"].notna() &
                (filtered["predicted_user_rating"] >= min_pred_score)
            ]
        elif pred_filter == "Unpredicted Only":
            filtered = filtered[filtered["predicted_user_rating"].isna()]
        elif pred_filter == "All" and min_pred_score > 1.0:
            filtered = filtered[
                filtered["predicted_user_rating"].isna() |
                (filtered["predicted_user_rating"] >= min_pred_score)
            ]

    # 4. Search Query (businessname, postcode, localauthorityname)
    if search_query:
        query = search_query.strip().lower()
        search_cols = [c for c in ["businessname", "BusinessName", "postcode", "PostCode", "localauthorityname", "LocalAuthorityName"] if c in filtered.columns]
        if search_cols:
            match_mask = pd.Series(False, index=filtered.index)
            for col in search_cols:
                match_mask = match_mask | filtered[col].astype(str).str.lower().str.contains(query, na=False)
            filtered = filtered[match_mask]

    # 5. Sorting
    if sort_by == "Predicted Rating (High to Low)" and "predicted_user_rating" in filtered.columns:
        filtered = filtered.sort_values(by="predicted_user_rating", ascending=False, na_position="last")
    elif sort_by == "User Rating (High to Low)" and "user_rating" in filtered.columns:
        filtered = filtered.sort_values(by="user_rating", ascending=False, na_position="last")
    elif sort_by == "First Seen (Newest)" and "first_seen" in filtered.columns:
        filtered = filtered.sort_values(by="first_seen", ascending=False, na_position="last")
    elif sort_by == "Business Name (A-Z)":
        name_col = "businessname" if "businessname" in filtered.columns else "BusinessName"
        if name_col in filtered.columns:
            filtered = filtered.sort_values(by=name_col, ascending=True, na_position="last")

    return filtered


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
        df_master = st.session_state.df_enriched.copy()
        
        # Top Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Loaded", len(df_master))
        m2.metric("In-Scope", len(df_master[df_master['in_scope'] == True]) if 'in_scope' in df_master.columns else 0)
        m3.metric("User Rated", len(df_master[df_master['user_rating'].notna()]) if 'user_rating' in df_master.columns else 0)
        m4.metric("ML Predicted", len(df_master[df_master['predicted_user_rating'].notna()]) if 'predicted_user_rating' in df_master.columns else 0)

        # --- In-Page Quick Filters & Slicers ---
        st.write("### 🔍 Explorer View & Filters")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            scope_view = st.selectbox(
                "Scope Filter",
                options=["All Loaded", "In-Scope (Restaurants)", "Out-of-Scope", "Unprocessed / Triage"],
                index=0,
                key="filter_scope_view"
            )
        with f_col2:
            rating_view = st.selectbox(
                "Rating Status",
                options=["All", "Unrated Only", "User Rated Only"],
                index=0,
                key="filter_rating_view"
            )
        with f_col3:
            pred_view = st.selectbox(
                "ML Prediction Status",
                options=["All", "Predicted Only", "Unpredicted Only"],
                index=0,
                key="filter_pred_view"
            )
        with f_col4:
            min_pred_score = st.slider(
                "Min Predicted Score",
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key="filter_min_pred_score"
            )

        f_search_col, f_sort_col = st.columns([2, 1])
        with f_search_col:
            search_query = st.text_input("🔎 Search by Business Name or Postcode", "", key="filter_search_text")
        with f_sort_col:
            sort_by = st.selectbox(
                "Sort By",
                options=[
                    "Predicted Rating (High to Low)",
                    "User Rating (High to Low)",
                    "First Seen (Newest)",
                    "Business Name (A-Z)",
                    "Natural / BQ Order"
                ],
                index=0,
                key="filter_sort_by"
            )

        # Filter & Sort Data
        df_filtered = filter_and_sort_restaurants(
            df_master,
            scope_filter=scope_view,
            rating_filter=rating_view,
            pred_filter=pred_view,
            min_pred_score=min_pred_score,
            search_query=search_query,
            sort_by=sort_by
        )

        st.caption(f"Showing **{len(df_filtered)}** of **{len(df_master)}** loaded restaurants.")

        # --- Master Interactive Table ---
        selection_event = display_data(df_filtered, key="master_grid")
        selected_rows = get_selected_rows(selection_event, df_filtered)
        num_selected = len(selected_rows) if (selected_rows is not None and not selected_rows.empty) else 0

        st.divider()

        # Selection Status Indicator
        if num_selected > 0:
            st.info(f"📌 **{num_selected} restaurant(s) selected** in the master table above. Choose an action sub-tab below to operate on them:")
        else:
            st.info("💡 Tip: Select one or more restaurants from the master table above to triage scope, assign ratings, or generate predictions.")

        # --- Action Sub-Tabs Under Table ---
        tab_triage, tab_rating, tab_predictions, tab_model = st.tabs([
            "📥 1. Scope Triage",
            "✍️ 2. Manual Rating",
            "🤖 3. ML Predictions",
            "⚙️ 4. Model Training"
        ])

        # -------------------------------------------------------------
        # SUB-TAB 1: SCOPE TRIAGE
        # -------------------------------------------------------------
        with tab_triage:
            st.subheader("Scope Triage")
            st.caption("Classify establishments into In-Scope (Restaurants) vs Out-of-Scope (Cafes, Bakeries, Supermarkets).")

            if num_selected > 0:
                col_map = {c.lower(): c for c in selected_rows.columns}
                id_col = col_map.get('fhrsid')
                
                st.write(f"**Batch Scope Triage for {num_selected} Selected Establishment(s):**")
                c_in, c_out, c_reset = st.columns(3)

                if c_in.button("✅ Mark as In-Scope (Restaurant)", key="btn_triage_in"):
                    if id_col:
                        ids = selected_rows[id_col].astype(str).tolist()
                        df_up = pd.DataFrame({'fhrsid': ids, 'in_scope': [True] * len(ids)})
                        with st.spinner(f"Updating {len(ids)} rows to In-Scope..."):
                            success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                            if success:
                                st.success(msg)
                                load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                st.rerun()
                            else:
                                st.error(msg)

                if c_out.button("🚫 Mark as Out-of-Scope (Bakery/Cafe)", key="btn_triage_out"):
                    if id_col:
                        ids = selected_rows[id_col].astype(str).tolist()
                        df_up = pd.DataFrame({'fhrsid': ids, 'in_scope': [False] * len(ids)})
                        with st.spinner(f"Updating {len(ids)} rows to Out-of-Scope..."):
                            success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                            if success:
                                st.success(msg)
                                load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                st.rerun()
                            else:
                                st.error(msg)

                if c_reset.button("🔄 Reset to Unprocessed", key="btn_triage_reset"):
                    if id_col:
                        ids = selected_rows[id_col].astype(str).tolist()
                        df_up = pd.DataFrame({'fhrsid': ids, 'in_scope': [None] * len(ids)})
                        with st.spinner(f"Resetting {len(ids)} rows..."):
                            success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                            if success:
                                st.success(msg)
                                load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                st.info("👆 Select one or more establishments in the table above to triage their scope.")

        # -------------------------------------------------------------
        # SUB-TAB 2: MANUAL RATING HUB
        # -------------------------------------------------------------
        with tab_rating:
            st.subheader("Manual Rating Hub (1 to 10 Scale)")
            st.caption("Assign user scores (1-10) and rating sources (desk evaluation or post-visit ground truth).")

            if num_selected > 0:
                col_map = {c.lower(): c for c in selected_rows.columns}
                id_col = col_map.get('fhrsid')

                # Section A: Quick Batch Score Tool
                st.write(f"#### ⚡ Quick-Apply Score to All {num_selected} Selected")
                qb_col1, qb_col2, qb_col3 = st.columns([1, 1, 2])
                with qb_col1:
                    batch_score = st.number_input(
                        "User Score (1-10)",
                        min_value=1,
                        max_value=10,
                        value=7,
                        step=1,
                        key="quick_score_input"
                    )
                with qb_col2:
                    batch_source = st.selectbox(
                        "Rating Source",
                        options=["desk", "visited"],
                        index=0,
                        key="quick_source_input"
                    )
                with qb_col3:
                    st.write("")
                    st.write("")
                    if st.button(f"⚡ Apply Score {batch_score} to All ({num_selected})", type="primary", key="btn_quick_apply_score"):
                        if id_col:
                            ids = selected_rows[id_col].astype(str).tolist()
                            df_up = pd.DataFrame({
                                'fhrsid': ids,
                                'user_rating': [int(batch_score)] * len(ids),
                                'rating_source': [str(batch_source)] * len(ids),
                                'in_scope': [True] * len(ids)
                            })
                            with st.spinner(f"Saving score {batch_score} for {len(ids)} restaurant(s)..."):
                                success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                                if success:
                                    st.success(f"Saved: {msg}")
                                    load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                    st.rerun()
                                else:
                                    st.error(msg)

                st.divider()

                # Section B: Interactive Data Editor for Individual Adjustments
                st.write(f"#### 📝 Individual Scores & Review Details")
                editor_df = selected_rows.copy()
                if 'user_rating' not in editor_df.columns:
                    editor_df['user_rating'] = pd.NA
                else:
                    editor_df['user_rating'] = pd.to_numeric(editor_df['user_rating'], errors='coerce')
                    
                if 'rating_source' not in editor_df.columns:
                    editor_df['rating_source'] = "desk"
                else:
                    editor_df['rating_source'] = editor_df['rating_source'].fillna("desk")
                
                display_cols = ['fhrsid', 'businessname', 'user_rating', 'rating_source', 'postcode', 'localauthorityname']
                available_cols = [c for c in display_cols if c in editor_df.columns]
                
                edited_df = st.data_editor(
                    editor_df[available_cols],
                    disabled=['fhrsid', 'businessname', 'postcode', 'localauthorityname'],
                    column_config={
                        "user_rating": st.column_config.NumberColumn(
                            "User Score (1-10)",
                            min_value=1,
                            max_value=10,
                            step=1,
                            required=True,
                            help="Rate from 1 to 10"
                        ),
                        "rating_source": st.column_config.SelectboxColumn(
                            "Rating Source",
                            options=["desk", "visited"],
                            required=True
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="rating_editor"
                )
                
                if st.button("💾 Submit Individual Scores to BigQuery", key="btn_submit_individual_scores"):
                    ed_col_map = {c.lower(): c for c in edited_df.columns}
                    ed_id_col = ed_col_map.get('fhrsid')
                    ed_score_col = ed_col_map.get('user_rating')
                    ed_source_col = ed_col_map.get('rating_source')
                    
                    if ed_id_col and ed_score_col and ed_source_col:
                        valid_rows = edited_df[edited_df[ed_score_col].notna()].copy()
                        if valid_rows.empty:
                            st.warning("Please enter a User Score (1-10) for at least one selected restaurant.")
                        else:
                            ids = valid_rows[ed_id_col].astype(str).tolist()
                            df_up = pd.DataFrame({
                                'fhrsid': ids,
                                'user_rating': valid_rows[ed_score_col].astype(int).tolist(),
                                'rating_source': valid_rows[ed_source_col].astype(str).tolist(),
                                'in_scope': [True] * len(ids)
                            })
                            with st.spinner(f"Saving scores for {len(ids)} restaurant(s)..."):
                                success, msg = bulk_update_reviews(project_id, dataset_id, table_id, df_up)
                                if success:
                                    st.success(f"Saved: {msg}")
                                    load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                                    st.rerun()
                                else:
                                    st.error(msg)
            else:
                st.info("👆 Select one or more establishments in the table above to assign manual scores.")

        # -------------------------------------------------------------
        # SUB-TAB 3: ML PREDICTIONS
        # -------------------------------------------------------------
        with tab_predictions:
            st.subheader("ML Predictions & Auto-Enrichment")
            st.caption("Generate preference ratings using BigQuery ML with automatic Maps & Gemini enrichment.")

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                force_maps = st.checkbox("Force Regenerate Maps Data", key="force_maps_unified")
            with col_opt2:
                force_gemini = st.checkbox("Force Regenerate Gemini Profiles", key="force_gemini_unified")

            if num_selected > 0:
                st.write(f"**Targeting {num_selected} Selected Restaurant(s):**")
                col_map = {c.lower(): c for c in selected_rows.columns}
                id_col = col_map.get('fhrsid')
                
                if st.button(f"⚡ Generate Predictions for {num_selected} Selected", type="primary", key="btn_gen_pred_selected"):
                    fhrsids = selected_rows[id_col].astype(str).tolist() if id_col else None
                    with st.spinner(f"Generating ML predictions for {num_selected} restaurants..."):
                        success, msg = generate_predictions(
                            project_id, dataset_id, table_id,
                            "restaurant_preference_model",
                            limit=len(fhrsids) if fhrsids else 50,
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
            else:
                st.write("**Batch Discovery Mode (No Selection):**")
                batch_limit = st.slider("Batch Size for Unrated Restaurants", min_value=10, max_value=100, value=50, step=10, key="batch_pred_limit")
                if st.button(f"⚡ Generate Batch Predictions (Top {batch_limit} Unrated)", type="primary", key="btn_gen_pred_batch"):
                    with st.spinner(f"Generating ML predictions for top {batch_limit} unrated restaurants..."):
                        success, msg = generate_predictions(
                            project_id, dataset_id, table_id,
                            "restaurant_preference_model",
                            limit=batch_limit,
                            target_fhrsids=None,
                            force_maps=force_maps,
                            force_gemini=force_gemini
                        )
                        if success:
                            st.success(msg)
                            load_data_into_state(project_id, dataset_id, table_id, in_scope_filter_values, outcode_filter, first_seen_start_date=first_seen_date, local_authority_filter=local_authority_filter)
                            st.rerun()
                        else:
                            st.error(msg)

        # -------------------------------------------------------------
        # SUB-TAB 4: MODEL TRAINING & OPERATIONS
        # -------------------------------------------------------------
        with tab_model:
            st.subheader("Train BQML Boosted Tree Regressor")
            st.caption("Trains continuous preference regression model using all in-scope rated restaurants (`user_rating` 1-10).")

            if "training_lock" not in st.session_state:
                st.session_state.training_lock = False
                
            if st.button("🚀 Train BQML Model (Async)", disabled=st.session_state.training_lock, key="btn_train_model_unified"):
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
