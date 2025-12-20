import streamlit as st
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import auth as firebase_auth
import json
import extra_streamlit_components as stx

class AuthManager:
    """
    Manages Firebase authentication state within a Streamlit application.
    """
    
    def __init__(self):
        # Initialize session state for user if it doesn't exist
        if 'user' not in st.session_state:
            st.session_state['user'] = None
        
        # Initialize Cookie Manager
        self.cookie_manager = stx.CookieManager()
        
        # Initialize Firebase Admin SDK if not already initialized
        if not firebase_admin._apps:
            try:
                firebase_admin.initialize_app()
            except Exception:
                pass

    def check_auth(self):
        """Checks for auth token in query params or cookies."""
        # 1. Check query params (from login popup redirect)
        params = st.query_params
        if 'token' in params:
            token = params['token']
            email = params.get('email')
            
            # Remove from URL
            new_params = dict(params)
            del new_params['token']
            if 'email' in new_params:
                del new_params['email']
            st.query_params.clear()
            for k, v in new_params.items():
                st.query_params[k] = v
            
            if token:
                user_data = {'email': email, 'token': token}
                st.session_state['user'] = user_data
                # Set cookie for persistence (1 day)
                self.cookie_manager.set('auth_user', json.dumps(user_data), key='set_auth_cookie')
                st.rerun()
        
        # 2. Check cookies (for returning users)
        if not st.session_state.get('user'):
            user_cookie = self.cookie_manager.get('auth_user')
            if user_cookie:
                try:
                    user_data = json.loads(user_cookie)
                    st.session_state['user'] = user_data
                except Exception:
                    pass

    def is_authenticated(self) -> bool:
        """Checks if the user is currently authenticated."""
        return st.session_state.get('user') is not None

    def get_user_email(self) -> Optional[str]:
        """Returns the authenticated user's email, or None."""
        user = st.session_state.get('user')
        if user:
            return user.get('email')
        return None

    def set_user(self, user_data: Dict[str, Any]):
        """Sets the user data in session state."""
        st.session_state['user'] = user_data

    def sign_out(self):
        """Clears the user data from session state and cookies."""
        st.session_state['user'] = None
        self.cookie_manager.delete('auth_user')

    def login_button(self):
        """Renders a Google Sign-In button using Firebase JS SDK."""
        config = st.secrets["firebase"]
        config_json = json.dumps(dict(config))
        
        html_code = f"""
        <div id="auth-container">
            <button id="login-button" style="
                background-color: white;
                color: #757575;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px 24px;
                font-size: 16px;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 10px;
                font-family: 'Roboto', sans-serif;
            ">
                <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" width="18px">
                Sign in with Google
            </button>
        </div>

        <script type="module">
            import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/9.22.1/firebase-app.js";
            import {{ getAuth, signInWithPopup, GoogleAuthProvider }} from "https://www.gstatic.com/firebasejs/9.22.1/firebase-auth.js";

            const firebaseConfig = {config_json};
            const app = initializeApp(firebaseConfig);
            const auth = getAuth(app);
            const provider = new GoogleAuthProvider();

            document.getElementById('login-button').addEventListener('click', () => {{
                signInWithPopup(auth, provider)
                    .then((result) => {{
                        const user = result.user;
                        user.getIdToken().then((idToken) => {{
                            const url = new URL(window.location.href);
                            url.searchParams.set('token', idToken);
                            url.searchParams.set('email', user.email);
                            window.parent.location.href = url.href;
                        }});
                    }}).catch((error) => {{
                        console.error("Auth error:", error);
                        alert("Authentication failed: " + error.message);
                    }});
            }});
        </script>
        """
        st.components.v1.html(html_code, height=100)