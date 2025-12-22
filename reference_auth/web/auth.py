import os
import firebase_admin
from firebase_admin import auth
from firebase_admin import credentials

# Initialize Firebase Admin SDK
# Note: On Google Cloud (Cloud Run/Functions), default credentials are used automatically.
# Locally, you might need to set GOOGLE_APPLICATION_CREDENTIALS or rely on gcloud auth application-default login.
try:
    firebase_admin.get_app()
except ValueError:
    firebase_project_id = os.environ.get("FIREBASE_PROJECT_ID")
    if firebase_project_id:
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {'projectId': firebase_project_id})
    else:
        # Fallback to default if not specified (though it should be for this scenario)
        firebase_admin.initialize_app()

def verify_token(id_token):
    """
    Verifies the Firebase ID token.
    Returns the decoded token dict if valid, else None.
    """
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Error verifying token: {e}")
        return None
