# Standard Library
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Third-party
import pandas as pd
import streamlit as st
from google.cloud import bigquery
import streamlit.components.v1 as components

# Local Modules
from api_client import fetch_api_data
from bq_utils import (
    update_rows_in_bigquery,
    sanitize_column_name,
    write_to_bigquery,
    load_all_data_from_bq,
    append_to_bigquery,
    execute_merge_query,
    BigQueryExecutionError,
    DataFrameConversionError,
    execute_gemini_enrichment,
    load_filtered_data_from_bq,
    bulk_update_reviews
)
from data_processing import load_json_from_local_file_path, load_master_data, process_and_update_master_data
from data_processing import load_data_from_csv
from auth.firebase_auth import AuthManager
from login import login_page

def display_data(data_to_display: List[Dict[str, Any]]):
    """
    Displays the given data using Streamlit, primarily as a Pandas DataFrame.
    """
    try:
        if not data_to_display:
            st.warning("No restaurant data to display.")
            return

        valid_items_for_df = [item for item in data_to_display if isinstance(item, dict)]
        
        if valid_items_for_df:
            df = pd.json_normalize(valid_items_for_df)
            st.dataframe(df)
        
    except Exception as e: 
        st.error(f"Error displaying DataFrame: {e}")

# Helper functions for handle_fetch_data_action
def _parse_coordinates(coordinate_pairs_str: str) -> List[tuple[float, float]]:
    valid_coords = []
    coordinate_lines = coordinate_pairs_str.strip().split('\n')
    for i, line in enumerate(coordinate_lines):
        line = line.strip()
        if not line: continue
        try:
            lon_str, lat_str = line.split(',')
            valid_coords.append((float(lon_str.strip()), float(lat_str.strip())))
        except ValueError:
            st.error(f"Error parsing coordinate line {i+1}: '{line}'.")
    return valid_coords

def _fetch_data_for_all_coordinates(valid_coords: List[tuple[float, float]], max_results: int) -> List[Dict[str, Any]]:
    all_api_establishments = []
    for lon, lat in valid_coords:
        page = 1
        while True:
            api_response = fetch_api_data(lon, lat, max_results, page)
            time.sleep(1) 
            if api_response:
                establishments = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
                if establishments is None: establishments = []
                all_api_establishments.extend(establishments)
                if len(establishments) < max_results: break
                page += 1
            else: break
    return all_api_establishments

def display_new_restaurants(new_restaurants: List[Dict[str, Any]]):
    if not new_restaurants: return
    st.subheader(f"Newly identified restaurants ({len(new_restaurants)})")
    df = pd.DataFrame(new_restaurants)
    st.dataframe(df, column_config={"Maps Link": st.column_config.LinkColumn("Research on Maps", display_text="Search Maps")}, hide_index=True)

def handle_fetch_data_action(coordinate_pairs_str: str, max_results: int, bq_full_path_str: str) -> List[Dict[str, Any]]:
    valid_coords = _parse_coordinates(coordinate_pairs_str)
    if not valid_coords: 
        st.error("No valid coordinates.")
        return []
    
    try:
        project_id, dataset_id, table_id = bq_full_path_str.split('.')
    except ValueError:
        st.error("Invalid BigQuery Path.")
        return []

    all_api_establishments = _fetch_data_for_all_coordinates(valid_coords, max_results)
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}

    master_restaurant_data = load_master_data(project_id, dataset_id, table_id, load_all_data_from_bq)
    new_restaurants = process_and_update_master_data(master_restaurant_data, combined_api_data)

    if new_restaurants:
        st.session_state.new_restaurants_to_review = new_restaurants
        st.success(f"Found {len(new_restaurants)} new restaurants!")
    
    display_data(master_restaurant_data)
    return master_restaurant_data

def auth_popup_handler():
    """
    Renders a dedicated page for handling the Firebase Auth Popup flow.
    """
    st.set_page_config(page_title="Authentication", layout="centered")
    
    config = st.secrets["firebase"]
    config_json = json.dumps(dict(config))

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Authenticating...</title>
        <style>
            body {{ font-family: sans-serif; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #f0f2f6; }}
            .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin-bottom: 20px; }}
            @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            .btn {{ background-color: #4285F4; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; }}
        </style>
    </head>
    <body>
        <div id="loading">
            <div class="spinner"></div>
            <p>Connecting to Google...</p>
        </div>
        <button id="auth-btn" class="btn" style="display:none;">Sign in with Google</button>

        <script type="module">
            import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/9.22.1/firebase-app.js";
            import {{ getAuth, signInWithPopup, GoogleAuthProvider }} from "https://www.gstatic.com/firebasejs/9.22.1/firebase-auth.js";

            const firebaseConfig = {config_json};
            const app = initializeApp(firebaseConfig);
            const auth = getAuth(app);
            const provider = new GoogleAuthProvider();

            function doAuth() {{
                signInWithPopup(auth, provider)
                    .then((result) => {{
                        result.user.getIdToken().then((idToken) => {{
                            if (window.opener) {{
                                window.opener.postMessage({{ 
                                    type: 'FIREBASE_AUTH_RESULT',
                                    success: true,
                                    data: {{ token: idToken, email: result.user.email }}
                                }}, '*');
                                window.close();
                            }}
                        }}));
                    }}).catch((error) => {{
                        console.error(error);
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('auth-btn').style.display = 'block';
                        alert("Auth failed: " + error.message);
                    }});
            }}

            document.getElementById('auth-btn').onclick = doAuth;
            doAuth();
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=600)

def main_ui():
    auth_manager = AuthManager()
    
    if st.query_params.get("mode") == "auth":
        auth_popup_handler()
        return

    token = st.query_params.get("token")
    if token:
        if auth_manager.verify_token(token):
            st.query_params.clear()
            st.rerun()

    if not auth_manager.is_authenticated():
        login_page(auth_manager)
        return

    st.title("Food Standards Agency API Explorer")

    user_email = auth_manager.get_user_email()
    st.sidebar.success(f"Logged in as: {user_email}")
    if st.sidebar.button("Sign Out"):
        auth_manager.sign_out()
        st.rerun()

    # Initialize session state variables
    if 'new_restaurants_to_review' not in st.session_state:
        st.session_state.new_restaurants_to_review = []

    st.subheader("Fetch API Data and Update Master List")
    coordinate_pairs_input = st.text_area("Enter longitude,latitude pairs (one per line):")
    max_results_input_ui = st.number_input("Enter Max Results", min_value=1, max_value=5000, value=200)
    bq_full_path_ui = st.text_input("Enter BigQuery Table Path (project.dataset.table)")

    if st.button("Fetch Data"):
        handle_fetch_data_action(coordinate_pairs_input, max_results_input_ui, bq_full_path_ui)

    if st.session_state.get('new_restaurants_to_review'):
        display_new_restaurants(st.session_state.new_restaurants_to_review)

    st.divider()
    st.subheader("Gemini Intelligence Analysis")
    col1, col2 = st.columns(2)
    with col1:
        connection_id_input = st.text_input("BigQuery Connection ID", value="eu.gemini")
    with col2:
        days_recent_input = st.number_input("Days Lookback", min_value=1, value=33)

    if st.button("Run Gemini Analysis"):
        if bq_full_path_ui:
            try:
                p, d, t = bq_full_path_ui.split('.')
                with st.spinner("Analyzing..."):
                    if execute_gemini_enrichment(p, d, t, connection_id_input, days_recent=days_recent_input):
                        st.success("Analysis Complete!")
            except ValueError: st.error("Invalid path.")

    st.divider()
    st.subheader("Export Filtered Data")
    with st.form("export_form"):
        c1, c2 = st.columns(2)
        with c1:
            export_days_input = st.number_input("Filter by 'First Seen' (days)", value=33, min_value=0)
        with c2:
            export_status_input = st.multiselect("Review Status", options=["pending", "not reviewed", "accepted", "rejected"], default=["pending", "not reviewed"])
        
        submitted = st.form_submit_button("Run Query & Preview")

    if submitted:
        if bq_full_path_ui:
            try:
                p, d, t = bq_full_path_ui.split('.')
                results = load_filtered_data_from_bq(p, d, t, days_filter=export_days_input, review_status_filter=export_status_input)
                if results:
                    st.dataframe(pd.DataFrame(results))
            except ValueError: st.error("Invalid path.")

    st.divider()
    st.subheader("Bulk Update Manual Reviews")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df_updates = pd.read_csv(uploaded_file)
        st.dataframe(df_updates.head())
        if st.button("Execute Bulk Update"):
            if bq_full_path_ui:
                try:
                    p, d, t = bq_full_path_ui.split('.')
                    if bulk_update_reviews(p, d, t, df_updates):
                        st.success("Bulk update successful!")
                except ValueError: st.error("Invalid path.")

if __name__ == "__main__":
    main_ui()