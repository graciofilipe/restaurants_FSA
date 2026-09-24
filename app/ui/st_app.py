import datetime

import streamlit as st
import pandas as pd
from app.services.bq_utils import (
    get_distinct_local_authorities,
    get_distinct_outcodes,
    load_filtered_data_from_bq,
    bulk_update_reviews,
)
from app.services.ml_prediction import generate_predictions
from app.core.data_processing import (
    enhance_dataframe_with_insights,
    calculate_restaurant_priority,
    get_outcode_coordinates,
)
from app.core.pillar_schema import ALL_COLUMNS as PILLAR_COLUMNS
from app.core.profile_freshness import count_needing_gemini_profile

st.set_page_config(page_title="FSA Restaurant Explorer", layout="wide")

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"

# The pillar half of this list is generated: it is the same 14 columns the
# profiler writes, in the order the prompt emits them. Before Phase 7 it was a
# hand-kept copy of a fourth naming convention -- flat, numeric-prefixed names
# that only ever existed in the DataFrame, built per row per rerun.
DISPLAY_COLUMNS = [
    "fhrsid", "businessname", "priority_score", "distance_km", "in_scope", "rating_source", "user_rating", "predicted_user_rating", "predicted_at",
    "addressline1", "addressline2", "addressline3",
    "postcode", "localauthorityname", "first_seen",
    "price_level", "maps_rating", "maps_reviews",
    "latitude", "longitude", "maps_url", "business_status", "website_url", "maps_types",
    "maps_found",
    *PILLAR_COLUMNS,
    "gemini_profiled_at",
    "gemini_insights_structured",
]

# The slicer vocabularies. Each sidebar selectbox is built from the same tuple
# `filter_and_sort_restaurants` compares against, so a renamed option cannot
# quietly stop matching. Before this, every filter accepted about three
# spellings of each answer and the widget passed a fourth literal -- the aliases
# were insurance against exactly the drift that sharing one name removes.
FILTER_ALL = "All"

SCOPE_ALL = "All Loaded"
SCOPE_IN = "In-Scope (Restaurants)"
SCOPE_OUT = "Out-of-Scope"
SCOPE_UNTRIAGED = "Unprocessed / Triage"
SCOPE_OPTIONS = (SCOPE_ALL, SCOPE_IN, SCOPE_OUT, SCOPE_UNTRIAGED)

RATED_YES = "Has User Rating (Rated)"
RATED_NO = "No User Rating (Unrated)"
RATED_OPTIONS = (FILTER_ALL, RATED_YES, RATED_NO)

PRED_YES = "Has Predicted Rating"
PRED_NO = "No Predicted Rating"
PRED_OPTIONS = (FILTER_ALL, PRED_YES, PRED_NO)

MATCH_YES = "Has Gemini Match Score"
MATCH_NO = "No Gemini Match Score"
MATCH_OPTIONS = (FILTER_ALL, MATCH_YES, MATCH_NO)

# The three answers `maps_found` can give. "No rating" was one answer covering
# two of them until Phase 8.
MAPS_ALL = FILTER_ALL
MAPS_FOUND = "Found on Google Maps"
MAPS_NOT_FOUND = "Not Found on Google Maps"
MAPS_NEVER_LOOKED_UP = "Not Looked Up Yet"
MAPS_OPTIONS = (MAPS_ALL, MAPS_FOUND, MAPS_NOT_FOUND, MAPS_NEVER_LOOKED_UP)

SORT_PRIORITY = "Priority Score (High to Low)"
SORT_DISTANCE = "Distance (Nearest First)"
SORT_PREDICTED = "Predicted Rating (High to Low)"
SORT_USER_RATING = "User Rating (High to Low)"
SORT_MAPS_RATING = "Maps Rating (High to Low)"
SORT_MATCH_SCORE = "Gemini Match Score (High to Low)"
SORT_FIRST_SEEN = "First Seen (Newest)"
SORT_NAME = "Business Name (A-Z)"
SORT_NATURAL = "Natural / BQ Order"
SORT_OPTIONS = (SORT_PRIORITY, SORT_DISTANCE, SORT_PREDICTED, SORT_USER_RATING,
                SORT_MAPS_RATING, SORT_MATCH_SCORE, SORT_FIRST_SEEN, SORT_NAME,
                SORT_NATURAL)

# Sort option -> (candidate columns, ascending). The first candidate the frame
# actually has wins; `SORT_NATURAL` is absent on purpose, since leaving the
# BigQuery order alone is what it means.
SORT_BY_COLUMN = {
    SORT_PRIORITY: (("priority_score",), False),
    SORT_DISTANCE: (("distance_km",), True),
    SORT_PREDICTED: (("predicted_user_rating",), False),
    SORT_USER_RATING: (("user_rating",), False),
    SORT_MAPS_RATING: (("maps_rating",), False),
    SORT_MATCH_SCORE: (("match_score",), False),
    SORT_FIRST_SEEN: (("first_seen",), False),
    SORT_NAME: (("businessname", "BusinessName"), True),
}

# Searched as one field. Both spellings, because the FSA API is PascalCase and
# BigQuery is lowercase and a loaded frame can carry either.
SEARCH_COLUMNS = ("businessname", "BusinessName", "postcode", "PostCode",
                  "localauthorityname", "LocalAuthorityName", "fhrsid", "FHRSID")


def _first_column(df: pd.DataFrame, names) -> str:
    """The first of `names` the frame actually has, or None."""
    return next((name for name in names if name in df.columns), None)

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

def reset_selection_state(key: str = "master_grid"):
    """Clears the master grid selection state from session state to avoid stale index issues."""
    if key in st.session_state:
        st.session_state.pop(key, None)

def get_selected_rows(event, df):
    """
    Safely retrieves the selected rows from the Streamlit dataframe selection event.
    Guards against out-of-bounds positional indices and non-DataFrame inputs.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if event and hasattr(event, "selection") and event.selection:
        rows = getattr(event.selection, "rows", None)
        if rows:
            valid_indices = [idx for idx in rows if isinstance(idx, int) and 0 <= idx < len(df)]
            if valid_indices:
                return df.iloc[valid_indices]
    return None


def filter_and_sort_restaurants(
    df: pd.DataFrame,
    scope_filter: str = SCOPE_ALL,
    user_rating_filter: str = FILTER_ALL,
    pred_rating_filter: str = FILTER_ALL,
    gemini_match_filter: str = FILTER_ALL,
    maps_filter: str = MAPS_ALL,
    min_pred_score: float = 1.0,
    search_query: str = "",
    sort_by: str = SORT_PREDICTED,
) -> pd.DataFrame:
    """
    Applies in-memory filtering and sorting to the restaurant DataFrame.

    Every filter takes one of the module's option constants. An unrecognised
    string is a no-op for that filter, which is how `FILTER_ALL` works.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # 1. Scope Filter
    if "in_scope" in filtered.columns:
        if scope_filter == SCOPE_IN:
            filtered = filtered[filtered["in_scope"] == True]  # noqa: E712 -- NULL must not match
        elif scope_filter == SCOPE_OUT:
            filtered = filtered[filtered["in_scope"] == False]  # noqa: E712
        elif scope_filter == SCOPE_UNTRIAGED:
            filtered = filtered[filtered["in_scope"].isna()]

    # 2. User Rating Filter
    if "user_rating" in filtered.columns:
        if user_rating_filter == RATED_YES:
            filtered = filtered[filtered["user_rating"].notna()]
        elif user_rating_filter == RATED_NO:
            filtered = filtered[filtered["user_rating"].isna()]

    # 3. ML Prediction Filter
    if "predicted_user_rating" in filtered.columns:
        if pred_rating_filter == PRED_YES:
            filtered = filtered[
                filtered["predicted_user_rating"].notna() &
                (filtered["predicted_user_rating"] >= min_pred_score)
            ]
        elif pred_rating_filter == PRED_NO:
            filtered = filtered[filtered["predicted_user_rating"].isna()]
        elif pred_rating_filter == FILTER_ALL and min_pred_score > 1.0:
            # An unpredicted row is not evidence of a low score, so the minimum
            # excludes only rows that have one and fall short.
            filtered = filtered[
                filtered["predicted_user_rating"].isna() |
                (filtered["predicted_user_rating"] >= min_pred_score)
            ]

    # 4. Gemini Match Score Filter
    # `insight_score` was the parser's alias for the same number; the column is
    # real now, so the alias is gone rather than carried as a fallback. A row
    # with raw JSON but no extracted score still counts as profiled.
    match_columns = [c for c in ("match_score", "gemini_insights_structured")
                     if c in filtered.columns]
    if match_columns and gemini_match_filter in (MATCH_YES, MATCH_NO):
        has_gemini = pd.Series(False, index=filtered.index)
        for column in match_columns:
            has_gemini = has_gemini | filtered[column].notna()
        filtered = filtered[has_gemini if gemini_match_filter == MATCH_YES else ~has_gemini]

    # 5. Google Maps Lookup Filter
    # `maps_found` answers this, not `maps_rating`. A NULL rating means either
    # "Places has no such restaurant" or "we have not asked yet", and those are
    # opposite answers: the first must never be re-queried, the second is the
    # whole enrichment backlog. A found restaurant with no rating is found.
    if "maps_found" in filtered.columns:
        if maps_filter == MAPS_FOUND:
            filtered = filtered[filtered["maps_found"] == True]  # noqa: E712 -- NULL must not match
        elif maps_filter == MAPS_NOT_FOUND:
            filtered = filtered[filtered["maps_found"] == False]  # noqa: E712
        elif maps_filter == MAPS_NEVER_LOOKED_UP:
            filtered = filtered[filtered["maps_found"].isna()]

    # 6. Search Query — one box across name, postcode, authority and FHRSID
    if search_query:
        query = search_query.strip().lower()
        search_cols = [c for c in SEARCH_COLUMNS if c in filtered.columns]
        if search_cols:
            match_mask = pd.Series(False, index=filtered.index)
            for col in search_cols:
                match_mask = match_mask | filtered[col].astype(str).str.lower().str.contains(query, na=False)
            filtered = filtered[match_mask]

    # 7. Sorting. Missing values sort last whichever direction is asked for:
    # an unscored restaurant is not the best one and not the worst one either.
    candidates, ascending = SORT_BY_COLUMN.get(sort_by, ((), None))
    sort_col = _first_column(filtered, candidates)
    if sort_col:
        filtered = filtered.sort_values(by=sort_col, ascending=ascending, na_position="last")

    return filtered


def set_enriched_frame(df):
    """The only way `df_enriched` is allowed to change.

    `data_version` is what `priority_for_current_frame` keys its memo on, so a
    frame swapped in behind its back would serve scores computed from the
    previous load. Routing every assignment through here means the bump cannot
    be forgotten at a new call site -- the same reasoning as
    `reset_selection_state`, which exists because Streamlit's positional row
    indices go stale the moment the frame underneath them changes.
    """
    st.session_state.df_enriched = df
    st.session_state.data_version = st.session_state.get('data_version', 0) + 1
    st.session_state.pop('priority_cache', None)


def priority_for_current_frame(df, anchor_lat, anchor_lon, weights):
    """`calculate_restaurant_priority`, memoized for the life of one frame.

    The ML Predictions tab re-scores the whole table on every rerun, and
    Streamlit reruns the script on every widget interaction anywhere on the
    page -- dragging the batch-size slider, ticking a checkbox in another tab.
    Nothing about those changes the scores, so the answer is the same one the
    previous rerun computed (D8).

    Not `@st.cache_data`: that hashes the DataFrame's contents to build its
    key, which for 11,268 rows is the same order of work as the scoring it
    would save. The version counter is O(1) and `set_enriched_frame` is the
    only thing that moves it.
    """
    key = (
        st.session_state.get('data_version'),
        float(anchor_lat), float(anchor_lon),
        tuple(sorted(weights.items())),
        datetime.date.today(),  # staleness tiers are measured against today
    )
    cached = st.session_state.get('priority_cache')
    if cached is not None and cached[0] == key:
        return cached[1].copy()

    scored = calculate_restaurant_priority(
        df, anchor_lat=anchor_lat, anchor_lon=anchor_lon, weights=weights)
    st.session_state['priority_cache'] = (key, scored)
    # A copy, so a caller that mutates what it got back cannot poison the memo
    # for the next rerun. This is what the uncached call always returned.
    return scored.copy()


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

                df_enriched = calculate_restaurant_priority(df_enriched)

                set_enriched_frame(df_enriched)
                st.session_state.data_loaded = True
                reset_selection_state()
            else:
                set_enriched_frame(pd.DataFrame())
                st.session_state.data_loaded = True
                reset_selection_state()
                st.warning("No data found matching criteria.")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")

def main():
    st.title("🍔 FSA Restaurant Explorer & Scoring Engine")
    
    if 'df_enriched' not in st.session_state:
        set_enriched_frame(pd.DataFrame())
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # One table, one dataset, one project -- not a setting. The sidebar used to
    # offer this as an editable box whose value was never read.
    bq_path = DEFAULT_BQ_PATH
    project_id, dataset_id, table_id = bq_path.split('.')

    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("📥 BigQuery Data Loader")
        
        scope_options_map = {
            "In Scope (Restaurants)": "in_scope",
            "Out of Scope (Cafes/Bakeries)": "out_of_scope",
            "Unprocessed / Needs Triage": "unprocessed"
        }
        
        selected_scope_labels = st.multiselect(
            "Establishment Scope (BQ Query)",
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

        first_seen_date = st.date_input(
            "First Seen After",
            value=None,
            max_value=pd.Timestamp.now().date(),
            help="Load restaurants first seen ON or AFTER this date."
        )

        if st.button("Load Data from BigQuery", type="primary", use_container_width=True):
            load_data_into_state(
                project_id, 
                dataset_id, 
                table_id, 
                in_scope_filter_values, 
                outcode_filter,
                first_seen_start_date=first_seen_date,
                local_authority_filter=local_authority_filter
            )

        st.divider()
        st.header("🎯 Master Table Slicers")
        st.caption("Instantly filter and slice loaded restaurants in real-time:")

        user_rating_slicer = st.selectbox(
            "User Rating",
            options=list(RATED_OPTIONS),
            index=0,
            key="slicer_user_rating"
        )

        pred_rating_slicer = st.selectbox(
            "ML Predicted Rating",
            options=list(PRED_OPTIONS),
            index=0,
            key="slicer_pred_rating"
        )

        min_pred_score = 1.0
        if pred_rating_slicer != PRED_NO:
            min_pred_score = st.slider(
                "Min Predicted Score",
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key="slicer_min_pred_score"
            )

        gemini_match_slicer = st.selectbox(
            "Gemini Match Score",
            options=list(MATCH_OPTIONS),
            index=0,
            key="slicer_gemini_match"
        )

        maps_slicer = st.selectbox(
            "Google Maps Lookup",
            options=list(MAPS_OPTIONS),
            index=0,
            key="slicer_maps_found",
            help="'Not Found' means Places was asked and had no match — those are not re-queried. 'Not Looked Up Yet' is the enrichment backlog."
        )

        scope_slicer = st.selectbox(
            "Scope View",
            options=list(SCOPE_OPTIONS),
            index=0,
            key="slicer_scope_view"
        )

        search_query = st.text_input("🔎 Quick Search", "", placeholder="Name, postcode, authority...", key="slicer_search_text")

        sort_by = st.selectbox(
            "Sort Order",
            options=list(SORT_OPTIONS),
            index=0,
            key="slicer_sort_by"
        )

    # --- Main Interface ---
    if st.session_state.data_loaded and not st.session_state.df_enriched.empty:
        df_master = st.session_state.df_enriched.copy()
        
        # Filter & Sort Data using Left-Panel Slicers
        df_filtered = filter_and_sort_restaurants(
            df_master,
            scope_filter=scope_slicer,
            user_rating_filter=user_rating_slicer,
            pred_rating_filter=pred_rating_slicer,
            gemini_match_filter=gemini_match_slicer,
            maps_filter=maps_slicer,
            min_pred_score=min_pred_score,
            search_query=search_query,
            sort_by=sort_by
        )

        # Top Summary Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Loaded", len(df_master))
        m2.metric("Filtered / Active", len(df_filtered))
        m3.metric("User Rated", len(df_master[df_master['user_rating'].notna()]) if 'user_rating' in df_master.columns else 0)
        m4.metric("ML Predicted", len(df_master[df_master['predicted_user_rating'].notna()]) if 'predicted_user_rating' in df_master.columns else 0)
        # Found on Maps, not "has a rating" -- a restaurant Places knows about
        # but nobody has rated is enriched, and counting it as missing is what
        # made the backlog look bigger than it is.
        m5.metric("Found on Maps", int((df_master['maps_found'] == True).sum()) if 'maps_found' in df_master.columns else 0)  # noqa: E712
        m6.metric("Gemini Evaluated",
                  len(df_master[df_master["match_score"].notna()]) if "match_score" in df_master.columns else 0)

        st.caption(f"Displaying **{len(df_filtered)}** of **{len(df_master)}** loaded restaurants.")

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
                st.write("#### 🎯 Smart Prioritized Batch Scoring (Budget Allocator)")
                st.caption("Heuristically prioritizes which restaurants to score or re-score to maximize Gemini API value and recommendation accuracy.")

                c_hq1, c_hq2, c_hq3 = st.columns([1, 1, 1])
                with c_hq1:
                    anchor_pc = st.text_input("📍 Anchor Postcode", value="SW16", key="anchor_postcode_input", help="Distance is measured from this London postcode.")
                    c_lat, c_lon = get_outcode_coordinates(anchor_pc)
                with c_hq2:
                    strategy_preset = st.selectbox(
                        "⚡ Strategy Preset",
                        options=[
                            "Balanced Active Discovery",
                            "Local Priority (SW16 & Nearby)",
                            "Model Refresh (Stale Re-scoring)",
                            "Top Maps Prior"
                        ],
                        index=0,
                        key="strategy_preset_select"
                    )
                with c_hq3:
                    target_mode = st.selectbox(
                        "🎯 Target Candidates",
                        options=["All (New & Stale Rescores)", "Unscored Candidates Only", "Stale Rescores Only"],
                        index=0,
                        key="target_mode_select"
                    )

                preset_weights = {
                    "Balanced Active Discovery": {"prox": 0.35, "stale": 0.35, "prior": 0.20, "scope": 0.10},
                    "Local Priority (SW16 & Nearby)": {"prox": 0.55, "stale": 0.25, "prior": 0.15, "scope": 0.05},
                    "Model Refresh (Stale Re-scoring)": {"prox": 0.25, "stale": 0.55, "prior": 0.15, "scope": 0.05},
                    "Top Maps Prior": {"prox": 0.30, "stale": 0.25, "prior": 0.40, "scope": 0.05},
                }
                active_weights = preset_weights.get(strategy_preset, preset_weights["Balanced Active Discovery"])

                # Recompute priorities based on current anchor & weights
                df_candidates = priority_for_current_frame(df_master, c_lat, c_lon, active_weights)

                # Filter candidate pool based on target_mode and scope (exclude confirmed out_of_scope)
                if "in_scope" in df_candidates.columns:
                    df_candidates = df_candidates[df_candidates["in_scope"] != False]

                if target_mode == "Unscored Candidates Only":
                    if "predicted_user_rating" in df_candidates.columns:
                        df_candidates = df_candidates[df_candidates["predicted_user_rating"].isna()]
                    if "user_rating" in df_candidates.columns:
                        df_candidates = df_candidates[df_candidates["user_rating"].isna()]
                elif target_mode == "Stale Rescores Only":
                    if "predicted_user_rating" in df_candidates.columns:
                        df_candidates = df_candidates[df_candidates["predicted_user_rating"].notna()]
                    if "user_rating" in df_candidates.columns:
                        df_candidates = df_candidates[df_candidates["user_rating"].isna()]

                df_ranked = df_candidates.sort_values(by="priority_score", ascending=False)

                batch_limit = st.slider("Batch Size (Budget of Restaurants to Score)", min_value=5, max_value=100, value=25, step=5, key="batch_pred_limit")

                top_candidates = df_ranked.head(batch_limit)
                num_candidates = len(top_candidates)

                if num_candidates > 0:
                    # The same predicate `generate_predictions` will apply, so
                    # the estimate cannot drift from the spend (D1). It counted
                    # the legacy V1 text column too -- NULL on every row, and
                    # since retired -- and knew nothing about staleness or the
                    # force checkbox.
                    gem_missing = count_needing_gemini_profile(top_candidates, force=force_gemini)

                    avg_dist = top_candidates["distance_km"].mean()
                    
                    st.info(f"📊 **Batch Queue:** {num_candidates} restaurants ranked | 💰 **Estimated New Gemini Calls:** {gem_missing} ({num_candidates - gem_missing} cached) | 📍 **Avg Distance:** {avg_dist:.1f} km from {anchor_pc.upper()}")
                    
                    # Preview table of top candidate restaurants
                    preview_cols = [c for c in ["fhrsid", "businessname", "priority_score", "distance_km", "staleness_score", "maps_rating", "predicted_user_rating", "predicted_at", "postcode"] if c in top_candidates.columns]
                    st.dataframe(top_candidates[preview_cols], hide_index=True, use_container_width=True)
                    
                    col_map = {c.lower(): c for c in top_candidates.columns}
                    id_col = col_map.get('fhrsid')
                    
                    if st.button(f"⚡ Score Top {num_candidates} Prioritized Restaurants", type="primary", key="btn_gen_pred_batch"):
                        target_ids = top_candidates[id_col].astype(str).tolist() if id_col else None
                        with st.spinner(f"Generating ML predictions for top {num_candidates} prioritized restaurants..."):
                            success, msg = generate_predictions(
                                project_id, dataset_id, table_id,
                                "restaurant_preference_model",
                                limit=num_candidates,
                                target_fhrsids=target_ids,
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
                    st.warning("No candidate restaurants found matching the selected target criteria.")

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
