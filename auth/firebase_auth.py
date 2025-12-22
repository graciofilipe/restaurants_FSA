import firebase_admin
from firebase_admin import auth, credentials
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self._initialize_firebase()

    def _initialize_firebase(self):
        """Initializes the Firebase Admin SDK if not already initialized."""
        try:
            # Check if already initialized
            firebase_admin.get_app()
        except ValueError:
            # Not initialized, so do it now
            try:
                # We use the project ID from secrets
                project_id = st.secrets["firebase"]["projectId"]
                # Initialize with default credentials (ADC) if possible, 
                # or just use the project ID for verification purposes.
                firebase_admin.initialize_app(options={'projectId': project_id})
                logger.info(f"Firebase Admin SDK initialized for project: {project_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
                st.error(f"Internal Auth Error: {e}")

    def verify_token(self, id_token: str) -> bool:
        """
        Verifies the Firebase ID token and sets the session state if valid.
        """
        try:
            decoded_token = auth.verify_id_token(id_token)
            email = decoded_token.get('email')
            
            if email:
                st.session_state['authenticated'] = True
                st.session_state['user_email'] = email
                logger.info(f"User {email} authenticated successfully.")
                return True
            else:
                logger.warning("Token verified but no email found in payload.")
                return False
                
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False

    def is_authenticated(self) -> bool:
        """Returns True if the user is currently authenticated."""
        return st.session_state.get('authenticated', False)

    def get_user_email(self) -> str | None:
        """Returns the authenticated user's email."""
        return st.session_state.get('user_email')

    def sign_out(self):
        """Clears the authentication state from the session."""
        if 'authenticated' in st.session_state:
            del st.session_state['authenticated']
        if 'user_email' in st.session_state:
            del st.session_state['user_email']
        logger.info("User signed out.")
