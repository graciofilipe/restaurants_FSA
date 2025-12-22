import streamlit as st
import streamlit.components.v1 as components
from auth.firebase_auth import AuthManager

def login_page(auth_manager: AuthManager):
    """
    Renders the login page with a link to the auth handler and a listener for the token.
    """
    st.set_page_config(page_title="Login - FSA API Explorer", layout="centered")
    
    st.markdown("""
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            margin-top: 10vh;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .login-title {
            margin-bottom: 1.5rem;
            color: #31333F;
            font-size: 2.5rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">FSA API Explorer</div>', unsafe_allow_html=True)
    st.write("Please sign in with your Google account to access the application.")
    
    # 1. Native Streamlit Link Button -> Opens /?mode=auth in new tab
    # We use the relative URL. 
    # Note: Streamlit Cloud/Run might strip query params if not careful, but usually works.
    # We use javascript to open it to ensure window.opener is set effectively for postMessage
    
    components.html("""
    <script>
    function openAuth() {
        // Calculate center for a nice popup experience (though it's a new window/tab)
        const width = 500;
        const height = 600;
        const left = (window.screen.width / 2) - (width / 2);
        const top = (window.screen.height / 2) - (height / 2);
        
        // Open the auth handler in a popup window
        window.open('/?mode=auth', 'firebaseAuthWindow', `width=${width},height=${height},top=${top},left=${left}`);
    }
    </script>
    <div style="display:flex; justify-content:center;">
        <button onclick="openAuth()" style="
            background-color: #4285F4;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            font-family: sans-serif;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            Sign in with Google
        </button>
    </div>
    """, height=80)

    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Listener Component
    # This invisible component listens for the postMessage from the popup
    components.html("""
    <script>
    window.addEventListener('message', (event) => {
        if (event.data.type === 'FIREBASE_AUTH_RESULT') {
            if (event.data.success) {
                const token = event.data.data.token;
                // Redirect parent to the same URL with token param
                const url = new URL(window.parent.location.href);
                url.searchParams.set('token', token);
                window.parent.location.href = url.toString();
            }
        }
    });
    </script>
    """, height=0)