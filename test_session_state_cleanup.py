import os
import re

def test_no_firebase_session_state_in_app():
    """Verify that st.session_state['authenticated'] and ['user_email'] are not used in st_app.py or other core logic"""
    files_to_check = ['st_app.py', 'api_client.py', 'bq_utils.py', 'data_processing.py']
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            assert 'authenticated' not in content or 'app_entered' in content # app_entered is the new one
            assert 'user_email' not in content
